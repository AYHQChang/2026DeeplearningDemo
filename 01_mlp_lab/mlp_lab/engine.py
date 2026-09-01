"""Experiment configuration, training loop and controlled comparisons."""


from dataclasses import dataclass, replace
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import DatasetBundle, make_dataset, resolve_device, seed_everything
from .model import MLP, _activation


@dataclass(frozen=True)
class ExperimentConfig:
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
        return self.test_accuracy[-1]


def validate_config(config: ExperimentConfig) -> None:
    if not config.hidden_sizes or any((size <= 0 for size in config.hidden_sizes)):
        raise ValueError('hidden_sizes 至少包含一个正整数。')
    if config.loss_name not in {'cross_entropy', 'mse'}:
        raise ValueError('loss_name 只能是 cross_entropy 或 mse。')
    if config.optimizer not in {'sgd', 'momentum', 'adam'}:
        raise ValueError('optimizer 只能是 sgd、momentum 或 adam。')
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError('dropout 必须位于 [0, 1) 区间。')
    if config.learning_rate <= 0 or config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError('learning_rate、epochs、batch_size 必须大于 0。')
    _activation(config.activation)


def _classification_loss(logits: torch.Tensor, labels: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == 'cross_entropy':
        return F.cross_entropy(logits, labels)
    if loss_name == 'mse':
        probabilities = torch.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(labels, num_classes=logits.shape[1]).float()
        return F.mse_loss(probabilities, one_hot_targets)
    raise ValueError(f'未知损失函数：{loss_name}')


def _make_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    if config.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    if config.optimizer == 'momentum':
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(x).argmax(dim=1)
    return float((predictions == y).float().mean().item())


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train_experiment(config: ExperimentConfig, data: DatasetBundle | None=None) -> TrainingResult:
    validate_config(config)
    device = resolve_device(config.device)
    seed_everything(config.seed)
    if data is None:
        data = make_dataset(config.dataset, n_samples=config.n_samples, noise=config.noise, seed=config.seed)
    model = MLP(hidden_sizes=config.hidden_sizes, activation=config.activation, dropout=config.dropout, n_classes=data.n_classes).to(device)
    optimizer = _make_optimizer(model, config)
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
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = _classification_loss(logits, batch_y, config.loss_name)
            loss.backward()
            squared_norm = sum((float(parameter.grad.detach().norm(2).item()) ** 2 for parameter in model.parameters() if parameter.grad is not None))
            batch_gradient_norms.append(squared_norm ** 0.5)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            total_examples += len(batch_x)
        train_loss.append(total_loss / total_examples)
        gradient_norm.append(float(np.mean(batch_gradient_norms)))
        train_accuracy.append(_accuracy(model, x_train, y_train))
        test_accuracy.append(_accuracy(model, x_test, y_test))
        if epoch in snapshot_epochs:
            snapshots[epoch] = _clone_state(model)
    _synchronize(device)
    elapsed = time.perf_counter() - start
    return TrainingResult(config=config, data=data, model=model, train_loss=train_loss, train_accuracy=train_accuracy, test_accuracy=test_accuracy, gradient_norm=gradient_norm, snapshots=snapshots, elapsed_seconds=elapsed, parameter_count=sum((parameter.numel() for parameter in model.parameters())))


def mode_defaults(mode: str) -> dict[str, int]:
    if mode == 'fast':
        return {'epochs': 80, 'n_samples': 600}
    if mode == 'extended':
        return {'epochs': 200, 'n_samples': 1200}
    raise ValueError('mode 只能是 fast 或 extended。')


def base_config(dataset: str='spiral', mode: str='fast', seed: int=42, device: str='auto') -> ExperimentConfig:
    defaults = mode_defaults(mode)
    dataset_noise = {'moons': 0.2, 'circles': 0.2, 'spiral': 0.45}
    if dataset not in dataset_noise:
        raise ValueError('dataset 只能是 moons、circles 或 spiral。')
    return ExperimentConfig(dataset=dataset, epochs=defaults['epochs'], n_samples=defaults['n_samples'], noise=dataset_noise[dataset], seed=seed, device=device)


def comparison_configs(kind: str, base: ExperimentConfig) -> list[ExperimentConfig]:
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
    data = make_dataset(base.dataset, n_samples=base.n_samples, noise=base.noise, seed=base.seed)
    return [train_experiment(config, data=data) for config in comparison_configs(kind, base)]
