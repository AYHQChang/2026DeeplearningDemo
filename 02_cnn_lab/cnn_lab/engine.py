"""小型 CNN 的配置、训练和池化对比。"""

from dataclasses import dataclass, replace
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import ImageDatasetBundle, load_digits_data, resolve_device, seed_everything
from .model import SmallCNN


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "基线：两层 CNN + Max Pooling"
    channels: tuple[int, int] = (8, 16)
    kernel_size: int = 3
    pooling: str = "max"
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
    if config.epochs <= 0 or config.batch_size <= 0 or config.learning_rate <= 0:
        raise ValueError("epochs、batch_size 和 learning_rate 必须大于 0。")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    data = data or load_digits_data(seed=config.seed)
    model = SmallCNN(
        image_size=data.image_size,
        n_classes=data.n_classes,
        channels=config.channels,
        kernel_size=config.kernel_size,
        pooling=config.pooling,
        dropout=config.dropout,
    ).to(device)
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
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
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
