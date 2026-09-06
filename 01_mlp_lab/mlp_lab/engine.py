"""分类 MLP 的配置、训练循环和单变量组件比较。"""


from dataclasses import dataclass, replace
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import DatasetBundle, make_dataset, resolve_device, seed_everything
from .model import MLP, _activation


# 共享服务器课堂上限：限制的是一次实验的计算量，不限制学生分析和写报告的时间。
MAX_EPOCHS = 200
MAX_SAMPLES = 1500
MAX_HIDDEN_LAYERS = 4
MAX_HIDDEN_WIDTH = 128
MAX_BATCH_SIZE = 512
MAX_UPDATE_STEPS = 12_000


@dataclass(frozen=True)
class ExperimentConfig:
    """一次实验的全部可调参数；冻结后可防止训练中被意外改写。"""
    name: str = '基线：ReLU + CrossEntropy + Adam'
    dataset: str = 'spiral'
    hidden_sizes: tuple[int, ...] = (32, 32)
    activation: str = 'relu'
    loss_name: str = 'cross_entropy'
    optimizer: str = 'adam'
    learning_rate: float = 0.02
    dropout: float = 0.0
    epochs: int = 80
    batch_size: int = 64
    n_samples: int = 600
    noise: float = 0.45
    seed: int = 42
    device: str = 'cpu'


@dataclass
class TrainingResult:
    """模型、数据、逐轮指标、阶段参数和运行信息的集合。"""
    config: ExperimentConfig
    data: DatasetBundle
    model: nn.Module
    train_loss: list[float]
    train_accuracy: list[float]
    test_accuracy: list[float]
    gradient_norm: list[float]
    snapshots: dict[int, dict[str, torch.Tensor]]
    elapsed_seconds: float
    parameter_count: int

    @property
    def final_test_accuracy(self) -> float:
        """返回最后一个 Epoch 记录的测试准确率。"""
        return self.test_accuracy[-1]


def estimate_update_steps(config: ExperimentConfig) -> int:
    """估算一次实验的参数更新次数，用于共享服务器计算预算检查。"""
    train_samples = config.n_samples - math.ceil(config.n_samples * 0.25)
    return math.ceil(train_samples / config.batch_size) * config.epochs


def validate_config(config: ExperimentConfig) -> None:
    """在训练前检查配置，并阻止误填参数拖慢共享服务器。"""
    if config.device == 'auto':
        raise ValueError('共享服务器不允许 device="auto"；请使用 cpu，或填写课堂分配的 cuda:编号。')
    if not config.hidden_sizes or any((size <= 0 for size in config.hidden_sizes)):
        raise ValueError('hidden_sizes 至少包含一个正整数。')
    if len(config.hidden_sizes) > MAX_HIDDEN_LAYERS or max(config.hidden_sizes) > MAX_HIDDEN_WIDTH:
        raise ValueError(
            f'共享服务器安全上限：隐藏层不超过 {MAX_HIDDEN_LAYERS} 层，'
            f'每层不超过 {MAX_HIDDEN_WIDTH} 个神经元。'
        )
    if config.loss_name not in {'cross_entropy', 'mse'}:
        raise ValueError('loss_name 只能是 cross_entropy 或 mse。')
    if config.optimizer not in {'sgd', 'momentum', 'adam'}:
        raise ValueError('optimizer 只能是 sgd、momentum 或 adam。')
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError('dropout 必须位于 [0, 1) 区间。')
    if config.learning_rate <= 0 or config.epochs <= 0 or config.batch_size <= 0 or config.n_samples <= 0:
        raise ValueError('learning_rate、epochs、batch_size、n_samples 必须大于 0。')
    if config.epochs > MAX_EPOCHS or config.n_samples > MAX_SAMPLES or config.batch_size > MAX_BATCH_SIZE:
        raise ValueError(
            f'共享服务器安全上限：epochs≤{MAX_EPOCHS}、n_samples≤{MAX_SAMPLES}、'
            f'batch_size≤{MAX_BATCH_SIZE}。四小时项目应增加分析深度，而不是扩大训练量。'
        )
    update_steps = estimate_update_steps(config)
    if update_steps > MAX_UPDATE_STEPS:
        raise ValueError(
            f'本次实验预计更新 {update_steps} 次，超过共享服务器上限 {MAX_UPDATE_STEPS} 次；'
            '请减少 epochs 或 n_samples，或适当增大 batch_size。'
        )
    _activation(config.activation)


def _classification_loss(logits: torch.Tensor, labels: torch.Tensor, loss_name: str) -> torch.Tensor:
    """计算分类损失；MSE 实验先把 logits 转成概率并把标签变成 one-hot。"""
    if loss_name == 'cross_entropy':
        return F.cross_entropy(logits, labels)
    if loss_name == 'mse':
        probabilities = torch.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(labels, num_classes=logits.shape[1]).float()
        return F.mse_loss(probabilities, one_hot_targets)
    raise ValueError(f'未知损失函数：{loss_name}')


def _make_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    """根据配置创建 SGD、带动量 SGD 或 Adam。"""
    if config.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    if config.optimizer == 'momentum':
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """在不计算梯度的情况下求一批数据的分类准确率。"""
    model.eval()
    with torch.no_grad():
        predictions = model(x).argmax(dim=1)
    return float((predictions == y).float().mean().item())


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """将当前参数独立复制到 CPU，供后续阶段图恢复。"""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _synchronize(device: torch.device) -> None:
    """计时时等待 CUDA 队列完成；CPU 不需要额外操作。"""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train_experiment(config: ExperimentConfig, data: DatasetBundle | None=None) -> TrainingResult:
    """训练一个分类 MLP，并记录曲线、梯度范数和阶段参数。

    输入特征形状为 [N,2]，标签为 [N]，模型输出 logits 为
    [B,n_classes]。传入 data 时可让多组组件实验共享完全相同的数据。
    """
    # 先检查配置、选择设备并固定随机状态，使错误尽早出现且实验可复现。
    validate_config(config)
    device = resolve_device(config.device)
    seed_everything(config.seed)
    # 单次实验自行生成数据；组件比较则从外部传入同一份数据。
    if data is None:
        data = make_dataset(config.dataset, n_samples=config.n_samples, noise=config.noise, seed=config.seed)
    model = MLP(hidden_sizes=config.hidden_sizes, activation=config.activation, dropout=config.dropout, n_classes=data.n_classes).to(device)
    optimizer = _make_optimizer(model, config)
    # DataLoader 的专用随机生成器保证批次打乱顺序可重复。
    # num_workers=0 避免共享服务器为每个 Notebook 创建加载子进程。
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(TensorDataset(data.x_train, data.y_train), batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)
    x_train = data.x_train.to(device)
    y_train = data.y_train.to(device)
    x_test = data.x_test.to(device)
    y_test = data.y_test.to(device)
    train_loss: list[float] = []
    train_accuracy: list[float] = []
    test_accuracy: list[float] = []
    gradient_norm: list[float] = []
    # 只保存少量有代表性的轮次，供决策边界阶段图使用。
    snapshot_epochs = {0, 1, min(5, config.epochs), max(1, config.epochs // 4), max(1, config.epochs // 2), config.epochs}
    snapshots = {0: _clone_state(model)}
    _synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        batch_gradient_norms: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)  # 清除上一批梯度。
            logits = model(batch_x)                # 前向：[B,2] → [B,C]。
            loss = _classification_loss(logits, batch_y, config.loss_name)
            loss.backward()                        # 反向计算每个参数的梯度。
            # 将所有参数梯度的 L2 范数组合为一个数，观察更新信号强弱。
            squared_norm = sum((float(parameter.grad.detach().norm(2).item()) ** 2 for parameter in model.parameters() if parameter.grad is not None))
            batch_gradient_norms.append(squared_norm ** 0.5)
            optimizer.step()                       # 优化器根据梯度更新参数。
            total_loss += float(loss.item()) * len(batch_x)
            total_examples += len(batch_x)
        train_loss.append(total_loss / total_examples)
        gradient_norm.append(float(np.mean(batch_gradient_norms)))
        # 每个 Epoch 结束后，用完整训练集和测试集记录学习曲线。
        train_accuracy.append(_accuracy(model, x_train, y_train))
        test_accuracy.append(_accuracy(model, x_test, y_test))
        if epoch in snapshot_epochs:
            snapshots[epoch] = _clone_state(model)
    _synchronize(device)
    elapsed = time.perf_counter() - start
    return TrainingResult(config=config, data=data, model=model, train_loss=train_loss, train_accuracy=train_accuracy, test_accuracy=test_accuracy, gradient_norm=gradient_norm, snapshots=snapshots, elapsed_seconds=elapsed, parameter_count=sum((parameter.numel() for parameter in model.parameters())))


def mode_defaults(mode: str) -> dict[str, int]:
    """返回课堂快速模式或课后扩展模式的轮数与样本数。"""
    if mode == 'fast':
        return {'epochs': 80, 'n_samples': 600}
    if mode == 'extended':
        return {'epochs': 200, 'n_samples': 1200}
    raise ValueError('mode 只能是 fast 或 extended。')


def base_config(dataset: str='spiral', mode: str='fast', seed: int=42, device: str='cpu') -> ExperimentConfig:
    """根据数据集与运行模式建立一份可继续 replace 的基线配置。"""
    defaults = mode_defaults(mode)
    dataset_noise = {'moons': 0.2, 'circles': 0.2, 'spiral': 0.45}
    if dataset not in dataset_noise:
        raise ValueError('dataset 只能是 moons、circles 或 spiral。')
    return ExperimentConfig(dataset=dataset, epochs=defaults['epochs'], n_samples=defaults['n_samples'], noise=dataset_noise[dataset], seed=seed, device=device)


def comparison_configs(kind: str, base: ExperimentConfig) -> list[ExperimentConfig]:
    """生成单变量对比配置；每组只改变指定组件。"""
    if kind == 'activation':
        return [replace(base, name=f'激活函数：{value}', activation=value) for value in ('relu', 'tanh', 'sigmoid')]
    if kind == 'depth':
        choices = ((32,), (32, 32), (32, 32, 32))
        return [replace(base, name=f'隐藏层：{len(value)} 层', hidden_sizes=value) for value in choices]
    if kind == 'loss':
        return [replace(base, name='损失：CrossEntropy', loss_name='cross_entropy'), replace(base, name='损失：MSE(概率 vs one-hot)', loss_name='mse')]
    if kind == 'optimizer':
        return [replace(base, name=f'优化器：{value}', optimizer=value) for value in ('sgd', 'momentum', 'adam')]
    if kind == 'dropout':
        return [replace(base, name=f'Dropout：{value:.1f}', dropout=value) for value in (0.0, 0.2, 0.5)]
    if kind == 'width':
        choices = ((8, 8), (32, 32), (64, 64))
        return [replace(base, name=f'隐藏宽度：{value[0]}', hidden_sizes=value) for value in choices]
    raise ValueError('experiment 只能是 activation、depth、loss、optimizer、dropout 或 width。')


def run_comparison(kind: str, base: ExperimentConfig) -> list[TrainingResult]:
    """让全部对比配置共享一份数据，并依次训练返回结果。"""
    data = make_dataset(base.dataset, n_samples=base.n_samples, noise=base.noise, seed=base.seed)
    return [train_experiment(config, data=data) for config in comparison_configs(kind, base)]
