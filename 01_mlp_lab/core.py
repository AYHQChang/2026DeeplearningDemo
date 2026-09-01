"""MLP 组件与二维决策边界实验室的共享核心代码。

这个文件解决四件事：

1. **准备数据**：离线生成月牙、同心圆和三分类螺旋，完成固定的训练/测试
   划分与标准化；每个样本的输入形状为 ``[2]``，标签是整数类别编号。
2. **定义模型与组件**：按配置创建任意隐藏层宽度的 MLP，并切换 ReLU、
   Tanh、Sigmoid、Cross-Entropy、MSE、SGD、Momentum、Adam 和 Dropout。
3. **记录训练过程**：保存每个 epoch 的 loss、训练/测试 accuracy、平均梯度
   L2 norm，并在若干关键 epoch 保存模型参数快照。
4. **生成课堂图形**：绘制数据画廊、决策边界演化、误判点、训练诊断图和
   单变量对比矩阵。所有图使用一致的类别颜色、坐标含义与评价指标。

``mlp_lab.ipynb`` 与 ``demo.py`` 只负责组织讲解和接收参数，模型与训练逻辑
全部集中在本文件中。这样修改一个组件时，Notebook 与脚本不会产生实现漂移。

本文件不访问网络、不下载数据，也不在导入时启动训练。主要张量约定如下：

- 输入 ``x``：``[batch_size, 2]``，两列为标准化后的二维特征；
- 标签 ``y``：``[batch_size]``，元素类型为 ``torch.long``；
- 模型输出 ``logits``：``[batch_size, n_classes]``，尚未经过 Softmax。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


# 三种类别颜色在所有图中保持不变，确保视觉变化只来自决策区域变化。
POINT_COLORS = ["#2563EB", "#DC2626", "#16A34A"]
REGION_COLORS = ListedColormap(["#BFDBFE", "#FECACA", "#BBF7D0"])


# =============================================================================
# 1. 实验配置与结果数据结构
# =============================================================================


@dataclass(frozen=True)
class ExperimentConfig:
    """描述一次可复现、可公平比较的 MLP 实验。

    Attributes:
        name: 图标题和结果表中显示的配置名称。
        dataset: ``moons``、``circles`` 或 ``spiral``。
        hidden_sizes: 每个隐藏层的神经元数量，例如 ``(32, 32)``。
        activation: ``relu``、``tanh`` 或 ``sigmoid``。
        loss_name: ``cross_entropy`` 或 ``mse``；两者都接收同一份 logits。
        optimizer: ``sgd``、``momentum`` 或 ``adam``。
        learning_rate: 优化器每次更新的步长。
        dropout: 隐藏层激活后的随机失活比例，范围为 ``[0, 1)``。
        epochs: 完整遍历训练集的次数。
        batch_size: 每次反向传播使用的样本数。
        n_samples: 生成数据集时使用的总样本数。
        noise: 数据生成噪声；含义因数据集而异。
        seed: 同时控制数据、参数初始化和 DataLoader shuffle 的随机种子。
        device: ``auto``、``cpu`` 或 ``cuda``。

    ``frozen=True`` 禁止训练中途原地修改字段。需要改变一个组件时使用
    ``dataclasses.replace`` 创建新配置，原配置继续作为公平对照的基准。
    """

    name: str = "基线：ReLU + CrossEntropy + Adam"
    dataset: str = "spiral"
    hidden_sizes: tuple[int, ...] = (32, 32)
    activation: str = "relu"
    loss_name: str = "cross_entropy"
    optimizer: str = "adam"
    learning_rate: float = 0.02
    dropout: float = 0.0
    epochs: int = 80
    batch_size: int = 64
    n_samples: int = 600
    noise: float = 0.45
    seed: int = 42
    device: str = "cpu"


@dataclass
class DatasetBundle:
    """保存同一次预处理得到的训练集、测试集与元数据。

    Attributes:
        x_train_raw: ``float32 [n_train, 2]`` NumPy 数组，尚未标准化。
        x_test_raw: ``float32 [n_test, 2]`` NumPy 数组，尚未标准化。
        x_train: ``float32`` 张量，形状 ``[n_train, 2]``。
        y_train: ``long`` 张量，形状 ``[n_train]``。
        x_test: ``float32`` 张量，形状 ``[n_test, 2]``。
        y_test: ``long`` 张量，形状 ``[n_test]``。
        n_classes: 类别总数；月牙/同心圆为 2，螺旋为 3。
        dataset_name: 数据集英文标识，用于打印和绘图。
        feature_mean: ``float64 [2]``，仅由训练集估计的标准化均值。
        feature_scale: ``float64 [2]``，仅由训练集估计的标准化尺度。

    四个张量默认保存在 CPU。训练函数会在需要时把批次移动到目标设备，
    从而让同一个 ``DatasetBundle`` 可以依次用于 CPU 与 GPU 对照。
    """

    x_train_raw: np.ndarray
    x_test_raw: np.ndarray
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    n_classes: int
    dataset_name: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray


@dataclass
class TrainingResult:
    """汇总一次训练的模型、逐轮曲线、阶段快照和效率指标。

    Attributes:
        config: 本次训练实际使用的不可变配置。
        data: 本次训练/测试使用的数据；用于保证绘图与训练对应。
        model: 已完成最后一个 epoch 的模型，位于所选运行设备上。
        train_loss: 每个 epoch 的平均训练损失。
        train_accuracy: 每个 epoch 结束后的完整训练集准确率。
        test_accuracy: 每个 epoch 结束后的完整测试集准确率。
        gradient_norm: 每个 epoch 内各 batch 总梯度 L2 norm 的平均值。
        snapshots: ``epoch -> state_dict``；张量统一复制到 CPU 以节省显存。
        elapsed_seconds: 纯训练循环耗时，不包含画图和数据准备。
        parameter_count: 模型所有可训练/不可训练参数元素总数。
    """

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
        """返回最后一个 epoch 的测试准确率，取值范围为 ``[0, 1]``。"""

        return self.test_accuracy[-1]


# =============================================================================
# 2. 运行环境、随机数与离线数据
# =============================================================================


def configure_chinese_font() -> None:
    """设置 Matplotlib 的中文字体候选与负号显示规则。

    Matplotlib 会按列表顺序选择本机存在的字体。Windows 通常命中
    ``Microsoft YaHei`` 或 ``SimHei``；其他平台可以退回 Noto/DejaVu。
    这个函数只修改当前 Python 进程的 ``rcParams``，不修改系统字体。

    Returns:
        ``None``。后续创建的图形自动使用该设置。
    """

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy、PyTorch CPU/CUDA 的随机数状态。

    Args:
        seed: 非负整数随机种子。相同种子用于数据生成、参数初始化和训练。

    Returns:
        ``None``。

    Notes:
        固定种子能够减少重复实验的波动，但 CPU 与 GPU 使用不同底层算子，
        因此不承诺两种设备最后一位小数完全一致。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """把设备字符串解析成 PyTorch ``torch.device``。

    Args:
        requested: ``auto``、``cpu`` 或 ``cuda``。``auto`` 优先 CUDA，
            检测不到 CUDA 时自动回退到 CPU。

    Returns:
        可以直接传给 ``Tensor.to`` 和 ``Module.to`` 的设备对象。

    Raises:
        ValueError: 输入不是三种支持值之一。
        RuntimeError: 显式要求 ``cuda``，但当前 PyTorch 检测不到 CUDA。

    ``auto`` 的回退是正常行为；显式 ``cuda`` 的失败则必须报错，避免误以为
    当前实验正在使用 GPU。
    """

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 只能是 auto、cpu 或 cuda。")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "当前 PyTorch 环境检测不到 CUDA。请改用 --device cpu，"
            "或课前安装与显卡驱动匹配的 CUDA 版 PyTorch。"
        )
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _make_spiral(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """生成三个互相缠绕的二维螺旋类别。

    Args:
        n_samples: 期望总样本数。为保证三类数量相同，实际数量向下取为
            ``3 * (n_samples // 3)``。
        noise: 加到角度上的高斯噪声标准差；数值越大，类别重叠越明显。
        seed: NumPy 随机数种子。

    Returns:
        ``(x, y)``：``x`` 为 ``float32 [n, 2]`` 坐标，``y`` 为
        ``int64 [n]`` 类别编号。该数据的弯曲边界适合观察 MLP 的非线性能力。
    """

    rng = np.random.default_rng(seed)
    n_classes = 3
    per_class = n_samples // n_classes
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for class_id in range(n_classes):
        radius = np.linspace(0.05, 1.0, per_class)
        angle = np.linspace(class_id * 4.0, (class_id + 1) * 4.0, per_class)
        angle += rng.normal(0.0, noise, per_class)
        points = np.column_stack((radius * np.sin(angle), radius * np.cos(angle)))
        xs.append(points)
        ys.append(np.full(per_class, class_id, dtype=np.int64))
    return np.vstack(xs).astype(np.float32), np.concatenate(ys)


def make_dataset(
    name: str,
    n_samples: int = 600,
    noise: float = 0.20,
    seed: int = 42,
) -> DatasetBundle:
    """离线生成二维数据，并完成分层划分与标准化。

    Args:
        name: ``moons``、``circles`` 或 ``spiral``。
        n_samples: 数据总量；默认 600，足以显示边界且训练很快。
        noise: 数据噪声。月牙使用坐标噪声，同心圆使用 ``noise * 0.55``，
            螺旋使用角度噪声。
        seed: 同时控制数据生成与训练/测试划分。

    Returns:
        ``DatasetBundle``。训练集占 75%，测试集占 25%；特征标准化参数只从
        训练集估计，避免测试集信息泄漏。所有张量初始位于 CPU。

    Raises:
        ValueError: 数据集名称不受支持。

    标准化把两个输入特征缩放到相近量级，使学习率或优化器对比不会被原始
    坐标尺度干扰。``stratify=y`` 保证训练集和测试集的类别比例一致。
    """

    if name == "moons":
        x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif name == "circles":
        x, y = make_circles(
            n_samples=n_samples,
            noise=noise * 0.55,
            factor=0.45,
            random_state=seed,
        )
    elif name == "spiral":
        x, y = _make_spiral(n_samples=n_samples, noise=noise, seed=seed)
    else:
        raise ValueError("dataset 只能是 moons、circles 或 spiral。")

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )
    x_train_raw = x_train_raw.astype(np.float32)
    x_test_raw = x_test_raw.astype(np.float32)
    scaler = StandardScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    return DatasetBundle(
        x_train_raw=x_train_raw,
        x_test_raw=x_test_raw,
        x_train=torch.from_numpy(x_train),
        y_train=torch.from_numpy(y_train).long(),
        x_test=torch.from_numpy(x_test),
        y_test=torch.from_numpy(y_test).long(),
        n_classes=int(np.max(y)) + 1,
        dataset_name=name,
        feature_mean=scaler.mean_.copy(),
        feature_scale=scaler.scale_.copy(),
    )


# =============================================================================
# 3. 可修改组件：激活函数、MLP 结构、损失函数与优化器
# =============================================================================


# ===== 可修改组件区开始 =====
# 首次扩展建议只改本区域；数据划分、训练记录和绘图可以保持不变。


def _activation(name: str) -> type[nn.Module]:
    """把配置字符串映射为激活层类型，而不是共享同一个层实例。

    Args:
        name: ``relu``、``tanh`` 或 ``sigmoid``。

    Returns:
        对应的 ``nn.Module`` 类；模型构造时会为每层创建独立实例。

    Raises:
        ValueError: 激活函数名称不受支持。
    """

    choices: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in choices:
        raise ValueError("activation 只能是 relu、tanh 或 sigmoid。")
    return choices[name]


class MLP(nn.Module):
    """用于二维分类的多层感知机（Multi-Layer Perceptron）。

    数据流为 ``[batch, 2] -> 多个隐藏层 -> [batch, n_classes]``。每个隐藏层
    依次包含 Linear、Activation，以及可选 Dropout。输出层只包含 Linear。

    最后一层直接输出 logits，不添加 Softmax。原因是 CrossEntropyLoss
    内部已经包含数值更稳定的 LogSoftmax；重复 Softmax 会改变损失含义。
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        activation: str,
        dropout: float,
        n_classes: int,
    ) -> None:
        """根据配置逐层构造 MLP。

        Args:
            hidden_sizes: 隐藏层宽度，例如 ``(32, 32)`` 表示两个隐藏层。
            activation: ``relu``、``tanh`` 或 ``sigmoid``。
            dropout: 每个隐藏层激活后的 Dropout 概率；0 表示不添加该层。
            n_classes: 输出类别数，也是 logits 的最后一维。
        """

        super().__init__()
        activation_class = _activation(activation)
        sizes = (2, *hidden_sizes, n_classes)
        layers: list[nn.Module] = []
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            is_output_layer = index == len(sizes) - 2
            if not is_output_layer:
                layers.append(activation_class())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行从二维特征到类别 logits 的前向传播。

        Args:
            x: ``float32 [batch_size, 2]`` 输入张量，需与模型位于同一设备。

        Returns:
            ``float32 [batch_size, n_classes]`` raw logits。调用者根据任务选择
            Cross-Entropy，或先 Softmax 后与 one-hot 目标计算 MSE。
        """

        return self.network(x)


def validate_config(config: ExperimentConfig) -> None:
    """在创建模型前检查所有会影响训练含义的配置。

    Args:
        config: 待检查的 ``ExperimentConfig``。

    Returns:
        ``None``；验证通过时不改变配置。

    Raises:
        ValueError: 隐藏层、损失函数、优化器、激活函数、Dropout 或正数参数
            不合法。提前失败比训练数分钟后产生 NaN 或错误结果更容易定位。
    """

    if not config.hidden_sizes or any(size <= 0 for size in config.hidden_sizes):
        raise ValueError("hidden_sizes 至少包含一个正整数。")
    if config.loss_name not in {"cross_entropy", "mse"}:
        raise ValueError("loss_name 只能是 cross_entropy 或 mse。")
    if config.optimizer not in {"sgd", "momentum", "adam"}:
        raise ValueError("optimizer 只能是 sgd、momentum 或 adam。")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout 必须位于 [0, 1) 区间。")
    if config.learning_rate <= 0 or config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("learning_rate、epochs、batch_size 必须大于 0。")
    _activation(config.activation)


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    """用合法的输出/目标组合计算分类损失。

    Args:
        logits: 模型原始输出，形状 ``[batch_size, n_classes]``。
        labels: 整数类别标签，形状 ``[batch_size]``。
        loss_name: ``cross_entropy`` 或 ``mse``。

    Returns:
        标量损失张量，保留计算图，可直接调用 ``backward()``。

    Raises:
        ValueError: 损失函数名称未知。

    Cross-Entropy 直接接收 logits 和整数标签。MSE 不是把类别编号当连续数回归，
    而是先把 logits 变成类别概率，再与 one-hot 目标比较。这使两种比较在数学上
    都成立，同时保留“合法方案不一定同样适合分类”的教学现象。
    """

    if loss_name == "cross_entropy":
        return F.cross_entropy(logits, labels)
    if loss_name == "mse":
        # MSE 比较的是“类别概率”和 one-hot 目标；不能把类别编号直接做回归。
        probabilities = torch.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(labels, num_classes=logits.shape[1]).float()
        return F.mse_loss(probabilities, one_hot_targets)
    raise ValueError(f"未知损失函数：{loss_name}")


def _make_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    """根据配置创建只管理当前模型参数的优化器。

    Args:
        model: 已构造并移动到目标设备的 MLP。
        config: 提供优化器名称与统一学习率的实验配置。

    Returns:
        ``torch.optim.Optimizer``。``momentum`` 使用 SGD 加 ``momentum=0.9``；
        三种优化器使用同一学习率，以满足一次只改变优化规则的对照要求。
    """

    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "momentum":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


# ===== 可修改组件区结束 =====


# =============================================================================
# 4. 训练循环、评价与过程记录
# =============================================================================


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """计算完整数据集上的分类准确率。

    Args:
        model: 输出 logits 的分类模型。
        x: ``[n_samples, 2]`` 特征，已位于模型设备。
        y: ``[n_samples]`` 整数标签，已位于模型设备。

    Returns:
        Python ``float``，范围 ``[0, 1]``。

    评价阶段调用 ``model.eval()`` 关闭 Dropout，并在 ``torch.no_grad()`` 中运行，
    防止评价图进入反向传播图并占用额外内存。
    """

    model.eval()
    with torch.no_grad():
        predictions = model(x).argmax(dim=1)
    return float((predictions == y).float().mean().item())


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """复制当前模型的完整 ``state_dict`` 到 CPU。

    Args:
        model: 任意 CPU 或 CUDA 上的 PyTorch 模型。

    Returns:
        新字典；键与 ``state_dict`` 相同，值是脱离计算图的 CPU 深拷贝。

    快照不能只保存 ``model.state_dict()`` 的引用，否则后续更新会同步改变旧快照。
    统一转到 CPU 也能避免六个阶段快照持续占用 GPU 显存。
    """

    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _synchronize(device: torch.device) -> None:
    """在 CUDA 计时边界等待异步算子完成；CPU 不需要操作。

    Args:
        device: 当前训练设备。

    Returns:
        ``None``。

    CUDA 默认异步提交内核。若不在计时前后同步，测得的时间可能只是提交耗时，
    无法与 CPU 训练时间进行有意义的比较。
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_experiment(
    config: ExperimentConfig,
    data: DatasetBundle | None = None,
) -> TrainingResult:
    """训练一次 MLP，并记录可解释训练过程的完整证据。

    Args:
        config: 模型、训练、数据规模、随机种子和设备设置。
        data: 可选的预生成 ``DatasetBundle``。组件对比必须传入同一个对象，
            从而保证数据点、划分与标准化完全相同；省略时按配置生成。

    Returns:
        ``TrainingResult``，包含最终模型、四条逐 epoch 曲线、关键阶段快照、
        参数量和同步后的训练耗时。

    Raises:
        ValueError: 配置或数据集名称不合法。
        RuntimeError: 显式请求的 CUDA 不可用。

    训练步骤固定为：清零梯度 → 前向传播 → 计算合法分类损失 → 反向传播 →
    记录总梯度范数 → 更新参数。每个 epoch 后在完整训练集和测试集上评价。
    """

    validate_config(config)
    device = resolve_device(config.device)
    seed_everything(config.seed)
    if data is None:
        data = make_dataset(
            config.dataset,
            n_samples=config.n_samples,
            noise=config.noise,
            seed=config.seed,
        )

    model = MLP(
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
        dropout=config.dropout,
        n_classes=data.n_classes,
    ).to(device)
    optimizer = _make_optimizer(model, config)

    # generator 固定每轮 shuffle 的随机顺序，使组件对比更公平。
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,  # Windows/Jupyter 下最稳妥，数据很小也无需多进程。
    )
    x_train = data.x_train.to(device)
    y_train = data.y_train.to(device)
    x_test = data.x_test.to(device)
    y_test = data.y_test.to(device)

    train_loss: list[float] = []
    train_accuracy: list[float] = []
    test_accuracy: list[float] = []
    gradient_norm: list[float] = []
    snapshot_epochs = {
        0,
        1,
        min(5, config.epochs),
        max(1, config.epochs // 4),
        max(1, config.epochs // 2),
        config.epochs,
    }
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

            # 所有参数梯度的 L2 范数反映本轮更新信号是否过大或过小。
            squared_norm = sum(
                float(parameter.grad.detach().norm(2).item()) ** 2
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            batch_gradient_norms.append(squared_norm**0.5)
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
    return TrainingResult(
        config=config,
        data=data,
        model=model,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        gradient_norm=gradient_norm,
        snapshots=snapshots,
        elapsed_seconds=elapsed,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def mode_defaults(mode: str) -> dict[str, int]:
    """返回快速运行与扩展运行的默认规模。

    Args:
        mode: ``fast`` 或 ``extended``。

    Returns:
        含 ``epochs`` 和 ``n_samples`` 的字典。fast 强调即时反馈；extended
        增加样本和轮数，用于观察更稳定的趋势。

    Raises:
        ValueError: 模式名称不受支持。
    """

    if mode == "fast":
        return {"epochs": 80, "n_samples": 600}
    if mode == "extended":
        return {"epochs": 200, "n_samples": 1200}
    raise ValueError("mode 只能是 fast 或 extended。")


# =============================================================================
# 5. 基线与单变量对照配置
# =============================================================================


def base_config(
    dataset: str = "spiral",
    mode: str = "fast",
    seed: int = 42,
    device: str = "auto",
) -> ExperimentConfig:
    """构造后续所有单变量对照共用的基线配置。

    Args:
        dataset: 二维数据名称。
        mode: ``fast`` 或 ``extended``，决定默认样本数和训练轮数。
        seed: 数据、初始化和 shuffle 的共同随机种子。
        device: ``auto``、``cpu`` 或 ``cuda``。

    Returns:
        默认使用 ``(32, 32)``、ReLU、Cross-Entropy、Adam 的不可变配置。
        后续使用 ``replace(base, target_component=...)`` 只修改一个组件。
    """

    defaults = mode_defaults(mode)
    dataset_noise = {"moons": 0.20, "circles": 0.20, "spiral": 0.45}
    if dataset not in dataset_noise:
        raise ValueError("dataset 只能是 moons、circles 或 spiral。")
    return ExperimentConfig(
        dataset=dataset,
        epochs=defaults["epochs"],
        n_samples=defaults["n_samples"],
        noise=dataset_noise[dataset],
        seed=seed,
        device=device,
    )


def comparison_configs(kind: str, base: ExperimentConfig) -> list[ExperimentConfig]:
    """从同一基线派生一个组件的全部对照配置。

    Args:
        kind: ``activation``、``depth``、``loss``、``optimizer``、``dropout``
            或 ``width``。
        base: 其余字段保持不变的基线配置。

    Returns:
        新配置列表。每个元素只修改目标组件及用于显示的 ``name`` 字段。

    Raises:
        ValueError: 对比类型不受支持。

    优化器对比保留相同学习率；这是“控制变量”设计，而不是为每个优化器寻找
    各自最优超参数。后者回答的是另一个问题，不能与本对照混为一谈。
    """

    if kind == "activation":
        return [replace(base, name=f"激活函数：{value}", activation=value) for value in ("relu", "tanh", "sigmoid")]
    if kind == "depth":
        choices = ((32,), (32, 32), (32, 32, 32))
        return [replace(base, name=f"隐藏层：{len(value)} 层", hidden_sizes=value) for value in choices]
    if kind == "loss":
        return [
            replace(base, name="损失：CrossEntropy", loss_name="cross_entropy"),
            replace(base, name="损失：MSE(概率 vs one-hot)", loss_name="mse"),
        ]
    if kind == "optimizer":
        return [replace(base, name=f"优化器：{value}", optimizer=value) for value in ("sgd", "momentum", "adam")]
    if kind == "dropout":
        return [replace(base, name=f"Dropout：{value:.1f}", dropout=value) for value in (0.0, 0.2, 0.5)]
    if kind == "width":
        choices = ((8, 8), (32, 32), (64, 64))
        return [replace(base, name=f"隐藏宽度：{value[0]}", hidden_sizes=value) for value in choices]
    raise ValueError("experiment 只能是 activation、depth、loss、optimizer、dropout 或 width。")


def run_comparison(kind: str, base: ExperimentConfig) -> list[TrainingResult]:
    """在完全相同的数据对象上训练一组单变量配置。

    Args:
        kind: 传给 ``comparison_configs`` 的组件名称。
        base: 数据规模、随机种子、设备和非目标组件的基线设置。

    Returns:
        顺序与配置列表一致的 ``TrainingResult`` 列表。

    数据只生成一次，然后把同一个 ``DatasetBundle`` 传给每次训练；这一步是
    公平比较的关键，避免不同随机划分掩盖组件本身的影响。
    """

    data = make_dataset(
        base.dataset,
        n_samples=base.n_samples,
        noise=base.noise,
        seed=base.seed,
    )
    return [train_experiment(config, data=data) for config in comparison_configs(kind, base)]


# =============================================================================
# 6. 二维决策区域计算与可视化
# =============================================================================


def _mesh(data: DatasetBundle, resolution: int = 180) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """创建覆盖全部样本的规则二维网格，用于绘制决策区域。

    Args:
        data: 提供坐标范围的训练/测试数据。
        resolution: 每个坐标轴的采样点数；最终预测点数为其平方。

    Returns:
        ``xx``、``yy`` 为 ``[resolution, resolution]`` NumPy 网格；``grid``
        为 ``float32 [resolution**2, 2]`` PyTorch 输入张量。

    坐标范围在观测点外增加 0.55 边距，防止边界贴着图框而难以判断形状。
    """

    all_x = torch.cat((data.x_train, data.x_test)).numpy()
    margin = 0.55
    x_min, x_max = all_x[:, 0].min() - margin, all_x[:, 0].max() + margin
    y_min, y_max = all_x[:, 1].min() - margin, all_x[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32))
    return xx, yy, grid


def _predict_grid(
    result: TrainingResult,
    grid: torch.Tensor,
    state: dict[str, torch.Tensor] | None = None,
) -> np.ndarray:
    """让最终模型或指定历史快照预测整个二维网格。

    Args:
        result: 提供模型与运行设备的训练结果。
        grid: ``float32 [n_grid_points, 2]`` CPU 网格。
        state: 可选的历史 CPU ``state_dict``；省略时使用最终模型。

    Returns:
        ``int64 [n_grid_points]`` NumPy 类别预测。

    若指定历史快照，函数会先保存最终状态，临时加载快照完成预测，再恢复最终
    状态。因此绘制早期边界不会永久改变 ``TrainingResult.model``。
    """
    device = next(result.model.parameters()).device
    current_state = _clone_state(result.model)
    if state is not None:
        result.model.load_state_dict(state)
    result.model.eval()
    with torch.no_grad():
        predictions = result.model(grid.to(device)).argmax(dim=1).cpu().numpy()
    if state is not None:
        result.model.load_state_dict(current_state)
    return predictions


def _draw_boundary(
    axis: plt.Axes,
    result: TrainingResult,
    state: dict[str, torch.Tensor] | None = None,
    mark_errors: bool = False,
) -> None:
    """在指定坐标轴绘制预测区域、训练点和可选测试误判点。

    Args:
        axis: 要绘制的 Matplotlib 坐标轴。
        result: 模型、数据和设备信息。
        state: 可选历史参数快照；用于展示某一 epoch 的边界。
        mark_errors: 是否叠加测试点。测试点用叉号表示，误判点更大、更粗。

    Returns:
        ``None``；图形直接添加到 ``axis``。

    背景浅色表示模型预测类别，实心圆表示训练样本。所有阶段复用同一数据坐标、
    类别颜色和网格分辨率，使边界变化可以直接进行视觉比较。
    """
    xx, yy, grid = _mesh(result.data)
    zz = _predict_grid(result, grid, state=state).reshape(xx.shape)
    axis.contourf(xx, yy, zz, levels=np.arange(result.data.n_classes + 1) - 0.5, cmap=REGION_COLORS, alpha=0.72)
    train_x = result.data.x_train.numpy()
    train_y = result.data.y_train.numpy()
    axis.scatter(
        train_x[:, 0],
        train_x[:, 1],
        c=[POINT_COLORS[value] for value in train_y],
        s=13,
        edgecolors="white",
        linewidths=0.25,
        alpha=0.72,
        label="训练样本",
    )
    if mark_errors:
        device = next(result.model.parameters()).device
        with torch.no_grad():
            predicted = result.model(result.data.x_test.to(device)).argmax(dim=1).cpu()
        wrong = predicted != result.data.y_test
        test_x = result.data.x_test.numpy()
        test_y = result.data.y_test.numpy()
        axis.scatter(
            test_x[:, 0],
            test_x[:, 1],
            c=[POINT_COLORS[value] for value in test_y],
            marker="x",
            s=np.where(wrong.numpy(), 50, 18),
            linewidths=np.where(wrong.numpy(), 1.8, 0.6),
            alpha=np.where(wrong.numpy(), 1.0, 0.35),
            label="测试样本（大叉为误判）",
        )
    # 使用普通数字而不是 Unicode 下标，避免部分 Windows 中文字体显示方框。
    axis.set_xlabel("标准化特征 x1")
    axis.set_ylabel("标准化特征 x2")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.12)


def plot_dataset_gallery(seed: int = 42, n_samples: int = 600) -> plt.Figure:
    """并排展示月牙、同心圆和螺旋三种二维数据。

    Args:
        seed: 三种数据共同使用的随机种子。
        n_samples: 每个数据集的总样本数。

    Returns:
        含三个等比例坐标轴的 ``matplotlib.figure.Figure``。函数不调用
        ``plt.show``，由 Notebook 或 ``save_or_show`` 决定显示/保存方式。

    这张图位于训练之前，用于先根据几何结构预测任务难度，而不是看到准确率后
    倒推解释。三图类别颜色含义保持一致。
    """

    configure_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    for axis, name, title in zip(
        axes,
        ("moons", "circles", "spiral"),
        ("月牙：局部弯曲", "同心圆：封闭边界", "螺旋：复杂非线性"),
    ):
        noise = {"moons": 0.20, "circles": 0.20, "spiral": 0.45}[name]
        data = make_dataset(name, n_samples=n_samples, noise=noise, seed=seed)
        points = torch.cat((data.x_train, data.x_test)).numpy()
        labels = torch.cat((data.y_train, data.y_test)).numpy()
        axis.scatter(points[:, 0], points[:, 1], c=[POINT_COLORS[value] for value in labels], s=18, alpha=0.8)
        axis.set_title(title)
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
    fig.suptitle("同一个 MLP，要面对三种不同难度的数据", fontsize=15)
    return fig


def plot_training_story(result: TrainingResult) -> plt.Figure:
    """把六个训练阶段与两类诊断信息组织成一张故事图。

    Args:
        result: 必须包含 Epoch 0、末轮及若干中间阶段快照的训练结果。

    Returns:
        ``2 x 4`` 布局的 Figure：前六格是相同坐标范围下的决策边界；第七格
        叠加 loss 与训练/测试 accuracy；第八格列出模型结构、参数量和耗时。

    该图回答“边界如何逐步形成”，适合基线实验。组件最终效果的横向比较应使用
    ``plot_comparison``，避免在一张图中混合时间演化和多模型比较。
    """

    configure_chinese_font()
    available_epochs = sorted(result.snapshots)
    stages = available_epochs[:2] + available_epochs[-4:]
    stages = list(dict.fromkeys(stages))[-6:]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), constrained_layout=True)
    flat_axes = axes.ravel()

    for axis, epoch in zip(flat_axes[:6], stages):
        _draw_boundary(axis, result, state=result.snapshots[epoch])
        axis.set_title(f"Epoch {epoch}：决策边界")

    epochs = np.arange(1, result.config.epochs + 1)
    curve_axis = flat_axes[6]
    curve_axis.plot(epochs, result.train_loss, color="#7C3AED", label="训练损失")
    curve_axis.set_xlabel("Epoch")
    curve_axis.set_ylabel("Loss")
    curve_axis.grid(alpha=0.2)
    accuracy_axis = curve_axis.twinx()
    accuracy_axis.plot(epochs, result.train_accuracy, color="#2563EB", label="训练准确率")
    accuracy_axis.plot(epochs, result.test_accuracy, color="#DC2626", linestyle="--", label="测试准确率")
    accuracy_axis.set_ylim(0.0, 1.02)
    accuracy_axis.set_ylabel("Accuracy")
    lines = curve_axis.lines + accuracy_axis.lines
    curve_axis.legend(lines, [line.get_label() for line in lines], loc="center right")
    curve_axis.set_title("训练是否收敛、是否过拟合？")

    info_axis = flat_axes[7]
    info_axis.axis("off")
    info_axis.text(
        0.02,
        0.98,
        "\n".join(
            (
                "最终实验卡",
                f"数据：{result.config.dataset}",
                f"隐藏层：{result.config.hidden_sizes}",
                f"激活：{result.config.activation}",
                f"损失：{result.config.loss_name}",
                f"优化器：{result.config.optimizer}",
                f"Dropout：{result.config.dropout}",
                f"参数量：{result.parameter_count:,}",
                f"测试准确率：{result.final_test_accuracy:.1%}",
                f"训练耗时：{result.elapsed_seconds:.2f} 秒",
            )
        ),
        va="top",
        fontsize=12,
        linespacing=1.55,
    )
    fig.suptitle(f"训练全过程：{result.config.name}", fontsize=16)
    return fig


def plot_final_diagnosis(result: TrainingResult) -> plt.Figure:
    """生成一次实验的四联最终诊断图。

    Args:
        result: 已训练模型及逐 epoch 记录。

    Returns:
        从左到右依次包含：最终决策边界与测试误判点、训练 loss、训练/测试
        accuracy、平均 gradient norm（对数轴）的 Figure。

    四个面板分别回答“学到了什么形状”“目标是否下降”“泛化差距多大”和
    “参数更新信号是否过强/过弱”，避免只看最终准确率下结论。
    """

    configure_chinese_font()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    _draw_boundary(axes[0], result, mark_errors=True)
    axes[0].set_title(f"最终边界｜测试 {result.final_test_accuracy:.1%}")
    axes[0].legend(fontsize=8, loc="upper right")

    epochs = np.arange(1, result.config.epochs + 1)
    axes[1].plot(epochs, result.train_loss, color="#7C3AED")
    axes[1].set_title("Loss：下降是否稳定？")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    axes[2].plot(epochs, result.train_accuracy, color="#2563EB", label="训练")
    axes[2].plot(epochs, result.test_accuracy, color="#DC2626", linestyle="--", label="测试")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title("Accuracy：泛化差距多大？")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()

    axes[3].plot(epochs, result.gradient_norm, color="#16A34A")
    axes[3].set_yscale("log")
    axes[3].set_title("Gradient norm：更新信号多强？")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("平均 L2 norm（对数轴）")
    for axis in axes[1:]:
        axis.grid(alpha=0.2)
    fig.suptitle(result.config.name, fontsize=15)
    return fig


def plot_comparison(results: list[TrainingResult], title: str) -> plt.Figure:
    """把一个组件的多个配置绘制为共享语义的对比矩阵。

    Args:
        results: 同一数据划分、同一随机种子下得到的非空结果列表。
        title: 图顶端显示的对比问题，例如“激活函数对比”。

    Returns:
        ``4 x n_configs`` 的 Figure。每列对应一个配置；四行依次是最终边界、
        loss、训练/测试 accuracy、gradient norm。准确率统一使用 ``[0, 1]``
        纵轴，梯度统一使用对数轴。

    Raises:
        ValueError: ``results`` 为空。

    按列组织后，只需从左向右扫视同一行，就能看到替换组件造成的结果差异；
    每列标题同时标出测试准确率、参数量与耗时。
    """

    if not results:
        raise ValueError("results 不能为空。")
    configure_chinese_font()
    n = len(results)
    fig, axes = plt.subplots(4, n, figsize=(5.0 * n, 15.2), squeeze=False, constrained_layout=True)

    for column, result in enumerate(results):
        _draw_boundary(axes[0, column], result, mark_errors=True)
        axes[0, column].set_title(
            f"{result.config.name}\n测试 {result.final_test_accuracy:.1%}｜"
            f"参数 {result.parameter_count:,}｜{result.elapsed_seconds:.2f}s"
        )
        epochs = np.arange(1, result.config.epochs + 1)
        axes[1, column].plot(epochs, result.train_loss, color="#7C3AED")
        axes[1, column].set_title("训练损失")
        axes[1, column].set_xlabel("Epoch")
        axes[1, column].set_ylabel("Loss")

        axes[2, column].plot(epochs, result.train_accuracy, color="#2563EB", label="训练")
        axes[2, column].plot(epochs, result.test_accuracy, color="#DC2626", linestyle="--", label="测试")
        axes[2, column].set_ylim(0.0, 1.02)
        axes[2, column].set_title("训练/测试准确率")
        axes[2, column].set_xlabel("Epoch")
        axes[2, column].set_ylabel("Accuracy")
        axes[2, column].legend()

        axes[3, column].plot(epochs, result.gradient_norm, color="#16A34A")
        axes[3, column].set_yscale("log")
        axes[3, column].set_title("平均梯度范数（对数轴）")
        axes[3, column].set_xlabel("Epoch")
        axes[3, column].set_ylabel("L2 norm")
        for row in (1, 2, 3):
            axes[row, column].grid(alpha=0.2)

    fig.suptitle(title + "｜每一列只替换一个目标组件", fontsize=17)
    return fig


def format_result_table(results: list[TrainingResult]) -> str:
    """把一个或多个训练结果格式化为统一宽度的纯文本指标表。

    Args:
        results: 一个或多个训练结果。

    Returns:
        可打印或写入日志文件的多行字符串。列包括最终测试准确率、参数量、
        训练耗时和末轮梯度范数；不创建 DataFrame，也不增加 pandas 依赖。
    """

    lines = [
        "配置对比",
        "-" * 86,
        f"{'配置':<30}{'测试准确率':>12}{'参数量':>12}{'耗时(秒)':>12}{'末轮梯度':>14}",
        "-" * 86,
    ]
    for result in results:
        lines.append(
            f"{result.config.name:<30}"
            f"{result.final_test_accuracy:>11.1%}"
            f"{result.parameter_count:>12,}"
            f"{result.elapsed_seconds:>12.2f}"
            f"{result.gradient_norm[-1]:>14.4g}"
        )
    lines.append("-" * 86)
    return "\n".join(lines)


def print_result_table(results: list[TrainingResult]) -> None:
    """在终端或 Notebook 输出与日志一致的纯文本指标表。

    Args:
        results: 一个或多个训练结果。

    Returns:
        ``None``。实际格式化由 ``format_result_table`` 完成，确保终端与 TXT
        日志不会因维护两份表格代码而出现列名或数值不一致。
    """

    print("\n" + format_result_table(results))


def save_or_show(
    figure: plt.Figure,
    filename: str,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """按调用参数保存、显示或关闭一张 Matplotlib 图。

    Args:
        figure: 已完成布局的 Figure。
        filename: 保存时使用的文件名，例如 ``comparison_activation.png``。
        save_dir: 可选目录；不存在时创建。省略则不保存文件。
        show: ``True`` 时调用 ``plt.show()``；``False`` 时关闭图，适合自动测试。

    Returns:
        ``None``。

    保存使用 150 DPI 和紧边界，兼顾投影清晰度与文件体积。显示与保存由入口层
    决定，绘图函数本身保持可组合，Notebook 与命令行可以复用同一张图。
    """

    if save_dir is not None:
        destination = Path(save_dir)
        destination.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination / filename, dpi=150, bbox_inches="tight")
        print(f"已保存图像：{destination / filename}")
    if show:
        plt.show()
    else:
        plt.close(figure)
