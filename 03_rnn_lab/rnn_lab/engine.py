"""延迟回忆实验的配置、训练、评估和三种循环单元对比。"""

from dataclasses import dataclass, replace
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    SequenceDatasetBundle,
    make_dataloaders,
    make_delayed_recall_data,
    resolve_device,
    seed_everything,
)
from .model import RecurrentClassifier


@dataclass(frozen=True)
class ExperimentConfig:
    """一轮受控实验的全部可修改参数。"""

    name: str = "基线：Vanilla RNN"
    cell_type: str = "rnn"
    sequence_length: int = 30
    hidden_size: int = 24
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 0.003
    batch_size: int = 64
    epochs: int = 20
    train_size: int = 900
    val_size: int = 240
    test_size: int = 240
    noise_scale: float = 0.5
    gradient_clip: float | None = 1.0
    seed: int = 42
    device: str = "cpu"


def validate_config(config: ExperimentConfig) -> None:
    """尽早报告常见配置错误，避免训练到一半才失败。"""

    if config.device in {"auto", "cuda"}:
        raise ValueError("共享服务器不允许 auto 或裸 cuda；请使用 cpu，或填写课堂分配的 cuda:编号。")
    if config.cell_type.lower() not in {"rnn", "lstm", "gru"}:
        raise ValueError("cell_type 只能是 rnn、lstm 或 gru。")
    if config.sequence_length < 3:
        raise ValueError("sequence_length 至少为 3。")
    if min(
        config.hidden_size,
        config.num_layers,
        config.learning_rate,
        config.batch_size,
        config.epochs,
        config.train_size,
        config.val_size,
        config.test_size,
    ) <= 0:
        raise ValueError("隐藏维度、层数、学习率、批量、轮数和数据量都必须大于 0。")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout 必须位于 [0, 1) 区间。")
    if config.gradient_clip is not None and config.gradient_clip <= 0:
        raise ValueError("gradient_clip 必须大于 0，或设为 None。")


def count_parameters(model: nn.Module) -> int:
    """统计需要训练的参数量。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_model(
    model: RecurrentClassifier,
    loaders: dict[str, DataLoader],
    config: ExperimentConfig,
) -> tuple[dict[str, list[float]], float]:
    """训练模型并记录 Loss、Accuracy 和裁剪前的梯度范数。"""

    device = resolve_device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "grad_norm": []}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    for _ in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        epoch_grad_norm = 0.0
        batches = 0
        for batch_x, batch_y in loaders["train"]:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            # clip_grad_norm_ 的返回值是裁剪前的总梯度范数，可同时用于诊断。
            max_norm = config.gradient_clip if config.gradient_clip is not None else float("inf")
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            total_examples += len(batch_x)
            epoch_grad_norm += float(grad_norm)
            batches += 1

        val_metrics = evaluate_model(model, loaders["val"], config.device)
        history["train_loss"].append(total_loss / total_examples)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["grad_norm"].append(epoch_grad_norm / batches)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return history, time.perf_counter() - start


def evaluate_model(
    model: RecurrentClassifier,
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, object]:
    """在一个数据划分上返回损失、准确率、预测和概率。"""

    resolved = resolve_device(device)
    model.to(resolved).eval()
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    losses = 0.0
    targets: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(resolved))
            losses += float(loss_function(logits, batch_y.to(resolved)).item())
            probabilities.append(logits.softmax(dim=1).cpu())
            targets.append(batch_y.cpu())
    all_targets = torch.cat(targets)
    all_probabilities = torch.cat(probabilities)
    predictions = all_probabilities.argmax(dim=1)
    return {
        "loss": losses / len(all_targets),
        "accuracy": float((predictions == all_targets).float().mean()),
        "targets": all_targets,
        "predictions": predictions,
        "probabilities": all_probabilities,
    }


def predict_proba(
    model: RecurrentClassifier,
    x: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """返回 CPU 上的分类概率，供 Notebook 展示单条预测。"""

    resolved = resolve_device(device)
    model.to(resolved).eval()
    with torch.no_grad():
        return model(x.to(resolved)).softmax(dim=1).cpu()


def input_gradient_by_time(
    model: RecurrentClassifier,
    x: torch.Tensor,
    target: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """计算一个样本的输入梯度范数，用于观察远处时间步的影响。"""

    resolved = resolve_device(device)
    model.to(resolved).eval()
    sample = x[:1].to(resolved).detach().requires_grad_(True)
    label = target[:1].to(resolved)
    model.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(model(sample), label).backward()
    return sample.grad.detach().norm(dim=2).squeeze(0).cpu()


def run_experiment(
    config: ExperimentConfig,
    data: SequenceDatasetBundle | None = None,
) -> dict[str, object]:
    """生成数据、训练并测试，返回 Notebook 需要的完整结果。"""

    validate_config(config)
    seed_everything(config.seed)
    data = data or make_delayed_recall_data(
        sequence_length=config.sequence_length,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed,
        noise_scale=config.noise_scale,
    )
    if data.sequence_length != config.sequence_length:
        raise ValueError("传入数据的序列长度与 config.sequence_length 不一致。")
    loaders = make_dataloaders(data, batch_size=config.batch_size, seed=config.seed)
    model = RecurrentClassifier(
        input_size=data.input_size,
        hidden_size=config.hidden_size,
        n_classes=data.n_classes,
        cell_type=config.cell_type,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    history, elapsed_seconds = train_model(model, loaders, config)
    metrics = evaluate_model(model, loaders["test"], config.device)
    return {
        "config": config,
        "data": data,
        "loaders": loaders,
        "model": model,
        "history": history,
        "metrics": metrics,
        "elapsed_seconds": elapsed_seconds,
        "parameter_count": count_parameters(model),
    }


def comparison_configs(
    base: ExperimentConfig,
    lengths: tuple[int, ...] = (12, 30, 60),
) -> list[ExperimentConfig]:
    """生成只改变循环单元和序列长度的受控实验配置。"""

    labels = {"rnn": "Vanilla RNN", "lstm": "LSTM", "gru": "GRU"}
    return [
        replace(
            base,
            name=f"{labels[cell_type]}｜长度 {length}",
            cell_type=cell_type,
            sequence_length=length,
        )
        for length in lengths
        for cell_type in labels
    ]


def compare_cells(
    base: ExperimentConfig,
    lengths: tuple[int, ...] = (12, 30, 60),
) -> list[dict[str, object]]:
    """按相同配置依次运行 RNN、LSTM 和 GRU。"""

    return [run_experiment(config) for config in comparison_configs(base, lengths)]
