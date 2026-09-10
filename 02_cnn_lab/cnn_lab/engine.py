"""小型 CNN 的配置、训练和池化对比。"""

from dataclasses import dataclass, replace
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import ImageDatasetBundle, load_digits_data, resolve_device, seed_everything
from .model import SmallCNN


MAX_EPOCHS = 40
MAX_CHANNELS = 64
MAX_KERNEL_SIZE = 7
MAX_BATCH_SIZE = 256
MAX_UPDATE_STEPS = 4_000
MAX_PARAMETERS = 750_000


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "基线：两层 CNN + Max Pooling"
    channels: tuple[int, int] = (8, 16)
    kernel_size: int = 3
    pooling: str = "max"
    activation: str = "relu"
    dropout: float = 0.0
    learning_rate: float = 0.01
    batch_size: int = 64
    epochs: int = 20
    seed: int = 42
    device: str = "cpu"


@dataclass
class TrainingResult:
    config: ExperimentConfig
    data: ImageDatasetBundle
    model: SmallCNN
    train_loss: list[float]
    train_accuracy: list[float]
    test_accuracy: list[float]
    elapsed_seconds: float
    parameter_count: int

    @property
    def final_test_accuracy(self) -> float:
        return self.test_accuracy[-1]


def estimate_update_steps(config: ExperimentConfig, train_samples: int) -> int:
    """计算一次 CNN 实验会执行多少次参数更新。"""
    return int(np.ceil(train_samples / config.batch_size)) * config.epochs


def validate_config(config: ExperimentConfig) -> None:
    """在创建模型前检查配置，避免误填参数拖慢共享服务器。"""
    if config.device in {"auto", "cuda"}:
        raise ValueError('共享服务器不允许 auto 或裸 cuda；请使用 cpu，或填写课堂分配的 cuda:编号。')
    if config.epochs <= 0 or config.batch_size <= 0 or config.learning_rate <= 0:
        raise ValueError("epochs、batch_size 和 learning_rate 必须大于 0。")
    if len(config.channels) != 2 or any(value <= 0 for value in config.channels):
        raise ValueError("channels 必须包含两个正整数。")
    if max(config.channels) > MAX_CHANNELS:
        raise ValueError(f"共享服务器安全上限：每层通道数不能超过 {MAX_CHANNELS}。")
    if config.kernel_size > MAX_KERNEL_SIZE:
        raise ValueError(f"共享服务器安全上限：kernel_size 不能超过 {MAX_KERNEL_SIZE}。")
    if config.epochs > MAX_EPOCHS or config.batch_size > MAX_BATCH_SIZE:
        raise ValueError(
            f"共享服务器安全上限：epochs≤{MAX_EPOCHS}、batch_size≤{MAX_BATCH_SIZE}。"
            "四小时项目应增加分析深度，而不是扩大训练量。"
        )


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(x.to(device)).argmax(dim=1).cpu()
    return float((predictions == y).float().mean().item())


def shift_images_right(images: torch.Tensor, pixels: int = 1) -> torch.Tensor:
    """向右平移并用 0 填充，不使用会从左侧绕回的 `torch.roll`。"""

    if pixels < 0 or pixels >= images.shape[-1]:
        raise ValueError("pixels 必须位于 0 到图片宽度减 1 之间。")
    shifted = torch.zeros_like(images)
    if pixels == 0:
        return images.clone()
    shifted[..., pixels:] = images[..., :-pixels]
    return shifted


def shifted_accuracy(result: TrainingResult, pixels: int = 1) -> float:
    device = next(result.model.parameters()).device
    return _accuracy(
        result.model, shift_images_right(result.data.x_test, pixels), result.data.y_test, device
    )


def train_experiment(
    config: ExperimentConfig, data: ImageDatasetBundle | None = None
) -> TrainingResult:
    validate_config(config)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    data = data or load_digits_data(seed=config.seed)
    update_steps = estimate_update_steps(config, len(data.x_train))
    if update_steps > MAX_UPDATE_STEPS:
        raise ValueError(
            f"本次实验预计更新 {update_steps} 次，超过共享服务器上限 {MAX_UPDATE_STEPS} 次；"
            "请减少 epochs，或适当增大 batch_size。"
        )
    model = SmallCNN(
        image_size=data.image_size,
        n_classes=data.n_classes,
        channels=config.channels,
        kernel_size=config.kernel_size,
        pooling=config.pooling,
        activation=config.activation,
        dropout=config.dropout,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > MAX_PARAMETERS:
        raise ValueError(
            f"本模型有 {parameter_count:,} 个参数，超过共享服务器上限 {MAX_PARAMETERS:,}；"
            "请减少通道数或使用 Pooling。"
        )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=0,
    )
    train_loss: list[float] = []
    train_accuracy: list[float] = []
    test_accuracy: list[float] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    for _ in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            total_examples += len(batch_x)
        train_loss.append(total_loss / total_examples)
        train_accuracy.append(_accuracy(model, data.x_train, data.y_train, device))
        test_accuracy.append(_accuracy(model, data.x_test, data.y_test, device))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return TrainingResult(
        config=config,
        data=data,
        model=model,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        elapsed_seconds=time.perf_counter() - start,
        parameter_count=parameter_count,
    )


def base_config(seed: int = 42, device: str = "cpu") -> ExperimentConfig:
    return ExperimentConfig(seed=seed, device=device)


def pooling_configs(base: ExperimentConfig) -> list[ExperimentConfig]:
    labels = {"max": "Max Pooling", "avg": "Average Pooling", "none": "不使用 Pooling"}
    return [replace(base, name=labels[value], pooling=value) for value in labels]


def compare_pooling_experiments(
    base: ExperimentConfig, data: ImageDatasetBundle | None = None
) -> list[TrainingResult]:
    data = data or load_digits_data(seed=base.seed)
    return [train_experiment(config, data=data) for config in pooling_configs(base)]
