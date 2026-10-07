"""小型 CNN 的配置、训练和池化对比。"""

from dataclasses import dataclass, replace
import csv
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import ImageDatasetBundle, load_digits_data, resolve_device, seed_everything
from .model import ComposableCNN, ConvBlock, SmallCNN


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
    blocks: tuple[ConvBlock, ...] | None = None


@dataclass
class TrainingResult:
    config: ExperimentConfig
    data: ImageDatasetBundle
    model: nn.Module
    train_loss: list[float]
    train_accuracy: list[float]
    test_accuracy: list[float]
    elapsed_seconds: float
    parameter_count: int
    val_accuracy: list[float] | None = None

    @property
    def final_test_accuracy(self) -> float:
        if not self.test_accuracy:
            raise ValueError("尚未进行最终测试；先按验证集选定模型，再调用 evaluate_final(result)。")
        return self.test_accuracy[-1]

    @property
    def final_validation_accuracy(self) -> float:
        if not self.val_accuracy:
            raise ValueError("当前实验没有验证集；探索模型时请提供 x_val 和 y_val。")
        return self.val_accuracy[-1]


def estimate_update_steps(config: ExperimentConfig, train_samples: int) -> int:
    """计算一次 CNN 实验会执行多少次参数更新。"""
    return int(np.ceil(train_samples / config.batch_size)) * config.epochs


def validate_config(config: ExperimentConfig) -> None:
    """在创建模型前检查配置，避免误填参数拖慢共享服务器。"""
    if config.device in {"auto", "cuda"}:
        raise ValueError('共享服务器不允许 auto 或裸 cuda；请使用 cpu，或填写课堂分配的 cuda:编号。')
    if config.epochs <= 0 or config.batch_size <= 0 or config.learning_rate <= 0:
        raise ValueError("epochs、batch_size 和 learning_rate 必须大于 0。")
    if config.blocks is None:
        if len(config.channels) != 2 or any(type(value) is not int or value <= 0 for value in config.channels):
            raise ValueError("channels 必须包含两个正整数。")
        if max(config.channels) > MAX_CHANNELS:
            raise ValueError(f"共享服务器安全上限：每层通道数不能超过 {MAX_CHANNELS}。")
        if config.kernel_size > MAX_KERNEL_SIZE:
            raise ValueError(f"共享服务器安全上限：kernel_size 不能超过 {MAX_KERNEL_SIZE}。")
    elif not 1 <= len(config.blocks) <= 3 or not all(isinstance(block, ConvBlock) for block in config.blocks):
        raise ValueError("blocks 必须包含 1–3 个 ConvBlock。")
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


def evaluation_data(
    result: TrainingResult, split: str = "auto"
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """探索阶段返回验证集；明确最终测试后才允许使用测试集诊断。"""
    if split == "auto":
        split = "test" if result.test_accuracy else "val"
    if split == "val":
        if result.data.x_val is None or result.data.y_val is None:
            raise ValueError("当前数据没有验证集。")
        return result.data.x_val, result.data.y_val, "验证"
    if split == "test":
        if not result.test_accuracy:
            raise ValueError("测试集尚未评估；先选定模型并调用 evaluate_final(result)。")
        return result.data.x_test, result.data.y_test, "测试"
    raise ValueError("split 只能是 auto、val 或 test。")


def shifted_accuracy(result: TrainingResult, pixels: int = 1, split: str = "auto") -> float:
    x, y, _ = evaluation_data(result, split=split)
    device = next(result.model.parameters()).device
    return _accuracy(result.model, shift_images_right(x, pixels), y, device)


def _check_split(x: torch.Tensor, y: torch.Tensor, data: ImageDatasetBundle, name: str) -> None:
    if x.ndim != 4 or x.shape[1:] != data.x_train.shape[1:] or len(x) == 0:
        raise ValueError(f"{name}图片应为非空 [N,C,H,W]，且与训练图片尺寸一致。")
    if x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name}图片应是有限数值的 float32 Tensor。")
    if y.ndim != 1 or len(y) != len(x) or y.dtype != torch.long:
        raise ValueError(f"{name}标签应为与图片数量一致的一维 long Tensor。")
    if int(y.min()) < 0 or int(y.max()) >= data.n_classes:
        raise ValueError(f"{name}标签必须位于 0 到 {data.n_classes - 1} 之间；请检查 CLASS_NAMES 和标签映射。")


def _check_output(model: nn.Module, data: ImageDatasetBundle, device: torch.device) -> None:
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.no_grad():
            sample = data.x_train[:2].to(device)
            output = model(sample)
        if not isinstance(output, torch.Tensor) or output.shape != (len(sample), data.n_classes):
            raise ValueError(f"模型输出应为 [B,{data.n_classes}] 的原始分数；请检查分类层类别数。")
    except RuntimeError as error:
        raise ValueError("模型无法接收当前图片；请检查输入通道、卷积后的尺寸和分类层输入维度。") from error
    finally:
        for module, training in modes.items():
            module.training = training


def evaluate_final(result: TrainingResult) -> float:
    """选定模型后计算一次测试成绩；重复调用返回已保存的成绩。"""
    if not result.test_accuracy:
        _check_split(result.data.x_test, result.data.y_test, result.data, "测试")
        device = next(result.model.parameters()).device
        result.test_accuracy.append(_accuracy(result.model, result.data.x_test, result.data.y_test, device))
    return result.final_test_accuracy


def train_experiment(
    config: ExperimentConfig, data: ImageDatasetBundle | None = None,
    model: nn.Module | None = None, *, evaluate_test: bool = True
) -> TrainingResult:
    """共用数据与网络训练入口。

    探索时提供验证集并用 evaluate_test=False，选定后调用 evaluate_final。
    直接传 model 时使用该实例；请在创建实例前设定 seed。
    evaluate_test=True 保留基础 Notebook 的运行方式。
    """
    validate_config(config)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    data = data or load_digits_data(seed=config.seed)
    if (data.x_val is None) != (data.y_val is None):
        raise ValueError("验证集图片和标签必须同时提供。")
    _check_split(data.x_train, data.y_train, data, "训练")
    if data.x_val is not None:
        _check_split(data.x_val, data.y_val, data, "验证")
    if not evaluate_test and data.x_val is None:
        raise ValueError("evaluate_test=False 需要验证集；数字实验可用 load_digits_data(val_size=0.2)。")
    update_steps = estimate_update_steps(config, len(data.x_train))
    if update_steps > MAX_UPDATE_STEPS:
        raise ValueError(
            f"本次实验预计更新 {update_steps} 次，超过共享服务器上限 {MAX_UPDATE_STEPS} 次；"
            "请减少 epochs，或适当增大 batch_size。"
        )
    if model is None:
        if config.blocks is not None:
            model = ComposableCNN(
                image_size=data.image_size, n_classes=data.n_classes,
                blocks=config.blocks, input_channels=data.channels, dropout=config.dropout,
            )
        else:
            if data.channels != 1:
                raise ValueError("SmallCNN 需要单通道图片；彩色数据请转灰度或使用 ComposableCNN。")
            model = SmallCNN(
                image_size=data.image_size, n_classes=data.n_classes,
                channels=config.channels, kernel_size=config.kernel_size,
                pooling=config.pooling, activation=config.activation, dropout=config.dropout,
            )
    if not isinstance(model, nn.Module):
        raise ValueError("model 必须是 PyTorch 的 nn.Module。")
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            if layer.out_channels > MAX_CHANNELS or max(layer.kernel_size) > MAX_KERNEL_SIZE:
                raise ValueError("共享服务器上限：卷积通道数≤64，kernel_size≤7。")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > MAX_PARAMETERS:
        raise ValueError(
            f"本模型有 {parameter_count:,} 个参数，超过共享服务器上限 {MAX_PARAMETERS:,}；"
            "请减少通道数或使用 Pooling。"
        )
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("模型需要至少一个可训练参数。")
    model = model.to(device)
    _check_output(model, data, device)
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
    val_accuracy: list[float] | None = [] if data.x_val is not None else None
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
        if val_accuracy is not None:
            val_accuracy.append(_accuracy(model, data.x_val, data.y_val, device))
        else:
            test_accuracy.append(_accuracy(model, data.x_test, data.y_test, device))

    if evaluate_test:
        _check_split(data.x_test, data.y_test, data, "测试")
        if val_accuracy is not None:
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
        val_accuracy=val_accuracy,
    )


def base_config(seed: int = 42, device: str = "cpu") -> ExperimentConfig:
    return ExperimentConfig(seed=seed, device=device)


def pooling_configs(base: ExperimentConfig) -> list[ExperimentConfig]:
    labels = {"max": "Max Pooling", "avg": "Average Pooling", "none": "不使用 Pooling"}
    return [
        replace(
            base, name=labels[value], pooling=value,
            blocks=(replace(base.blocks[0], pooling=value), *base.blocks[1:]) if base.blocks else None,
        )
        for value in labels
    ]


def compare_pooling_experiments(
    base: ExperimentConfig, data: ImageDatasetBundle | None = None, *, evaluate_test: bool = True
) -> list[TrainingResult]:
    data = data or load_digits_data(seed=base.seed)
    return [train_experiment(config, data=data, evaluate_test=evaluate_test) for config in pooling_configs(base)]


def experiment_records(results: list[TrainingResult]) -> list[dict]:
    """记录实际训练的网络及设置；未最终测试的试验保留空测试成绩。"""
    records = []
    for result in results:
        config = result.config
        if isinstance(result.model, ComposableCNN):
            structure = " | ".join(
                f"C{block.out_channels}/K{block.kernel_size}/{block.activation}/"
                f"pool={block.pooling}/BN={block.batch_norm}/drop={block.dropout}"
                for block in result.model.specs
            ) + f" | head_drop={config.dropout}"
        elif isinstance(result.model, SmallCNN):
            structure = f"SmallCNN channels={config.channels}, kernel={config.kernel_size}, pooling={config.pooling}, activation={config.activation}, dropout={config.dropout}"
        else:
            structure = repr(result.model)
        records.append({
            "name": config.name, "dataset": result.data.dataset_name,
            "structure": structure, "parameters": result.parameter_count,
            "learning_rate": config.learning_rate, "batch_size": config.batch_size,
            "epochs": config.epochs, "seed": config.seed, "device": config.device,
            "update_steps": estimate_update_steps(config, len(result.data.x_train)),
            "train_accuracy": result.train_accuracy[-1],
            "validation_accuracy": result.val_accuracy[-1] if result.val_accuracy else None,
            "test_accuracy": result.test_accuracy[-1] if result.test_accuracy else None,
        })
    return records


def save_experiment_records(results: list[TrainingResult], path: str | Path) -> Path:
    """保存可直接用 Excel 打开的 UTF-8 CSV，便于把实验记录纳入报告。"""
    records = experiment_records(results)
    if not records:
        raise ValueError("请先训练至少一个实验，再保存记录。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path
