"""延迟回忆实验的配置、训练、评估和三种循环单元对比。"""

import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path
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
from .observation import evaluating


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
    if config.sequence_length < 1:
        raise ValueError("sequence_length 必须大于 0；默认延迟回忆数据另要求至少 3 步。")
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
    if config.noise_scale < 0:
        raise ValueError("noise_scale 不能小于 0。")


def _check_split(x: torch.Tensor, y: torch.Tensor, data: SequenceDatasetBundle, name: str) -> None:
    if x.ndim != 3 or min(x.shape) < 1:
        raise ValueError(f"{name}输入必须是非空 [N,T,F] 张量；请先完成图片转序列 TODO。")
    if tuple(x.shape[1:]) != (data.sequence_length, data.input_size):
        raise ValueError(f"{name}的 T/F 必须与训练集一致。")
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name}输入须为有限浮点数。")
    if y.ndim != 1 or len(y) != len(x) or y.dtype != torch.int64:
        raise ValueError(f"{name}标签须为 torch.int64 的 [N] 张量，并与输入一一对应。")
    if data.n_classes < 2 or int(y.min()) < 0 or int(y.max()) >= data.n_classes:
        raise ValueError(f"{name}标签须位于 0 到 len(class_names)-1，且至少有两个类别。")


def count_parameters(model: nn.Module) -> int:
    """统计需要训练的参数量。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_model(
    model: nn.Module,
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
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, object]:
    """在一个数据划分上返回损失、准确率、预测和概率。"""

    resolved = resolve_device(device)
    model.to(resolved)
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    losses = 0.0
    targets: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    with evaluating(model), torch.no_grad():
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
    model: nn.Module,
    x: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """返回 CPU 上的分类概率，供 Notebook 展示单条预测。"""

    resolved = resolve_device(device)
    model.to(resolved)
    with evaluating(model), torch.no_grad():
        return model(x.to(resolved)).softmax(dim=1).cpu()


def input_gradient_by_time(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """计算一个样本的输入梯度范数，用于观察远处时间步的影响。"""

    # 观察跟随模型实际设备；保留 device 形参以兼容旧 Notebook，不搬动模型。
    resolved = next(model.parameters()).device
    sample = x[:1].to(resolved).detach().requires_grad_(True)
    label = target[:1].to(resolved)
    # autograd.grad 只返回输入梯度，不清空或累加模型已有的参数梯度。
    # CUDA 的 cuDNN 循环层不支持 eval 模式反传；观察时使用原生实现。
    with evaluating(model), torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        loss = nn.CrossEntropyLoss()(model(sample), label)
        gradient, = torch.autograd.grad(loss, sample)
    return gradient.detach().norm(dim=2).squeeze(0).cpu()


def run_experiment(
    config: ExperimentConfig,
    data: SequenceDatasetBundle | None = None,
    model: nn.Module | None = None,
    *,
    evaluate_test: bool = True,
) -> dict[str, object]:
    """训练并记录验证成绩；探索时显式指定 evaluate_test=False。

    自定义 model 直接使用传入实例；请在创建实例之前 seed_everything(config.seed)。
    默认测试行为仅为兼容旧入口；按验证集选定后调用 evaluate_final(result)。
    """

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
    _check_split(data.x_train, data.y_train, data, "训练集")
    _check_split(data.x_val, data.y_val, data, "验证集")
    loaders = make_dataloaders(data, batch_size=config.batch_size, seed=config.seed)
    if model is None:
        model = RecurrentClassifier(
            input_size=data.input_size,
            hidden_size=config.hidden_size,
            n_classes=data.n_classes,
            cell_type=config.cell_type,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    device = resolve_device(config.device)
    model.to(device)
    with evaluating(model), torch.no_grad():
        example = data.x_train[:2].to(device)
        output = model(example)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (len(example), data.n_classes):
            raise ValueError("模型 forward 必须返回 [B,K] logits；K 应为数据的类别数。")
        if not bool(torch.isfinite(output).all()):
            raise ValueError("模型输出包含非有限值，请检查 forward。")
    history, elapsed_seconds = train_model(model, loaders, config)
    result = {
        "config": config,
        "data": data,
        "loaders": loaders,
        "model": model,
        "history": history,
        "val_metrics": evaluate_model(model, loaders["val"], config.device),
        "metrics": None,
        "elapsed_seconds": elapsed_seconds,
        "parameter_count": count_parameters(model),
    }
    if evaluate_test:
        evaluate_final(result)
    return result


def evaluate_final(result: dict[str, object]) -> dict[str, object]:
    """对已选定、不再调参的模型进行最终测试；重复调用返回缓存成绩。

    此函数不重新训练、不改变 requires_grad。选定后应停止修改该模型，
    缓存记录的是第一次最终测试的模型成绩，不能把缓存当作新训练模型的结果。
    """

    if result["metrics"] is None:
        data = result["data"]
        _check_split(data.x_test, data.y_test, data, "测试集")
        result["metrics"] = evaluate_model(result["model"], result["loaders"]["test"], result["config"].device)
    return result["metrics"]


def experiment_records(results: list[dict[str, object]]) -> list[dict]:
    """导出配置和实际尺寸；未最终测试的实验保留空白测试成绩。"""

    records = []
    for result in results:
        record = asdict(result["config"])
        data = result["data"]
        recurrent = getattr(result["model"], "recurrent", None)
        record.update({
            "model_class": type(result["model"]).__name__,
            "actual_recurrent_type": type(recurrent).__name__ if recurrent is not None else "",
            "actual_hidden_size": getattr(recurrent, "hidden_size", None),
            "actual_num_layers": getattr(recurrent, "num_layers", None),
            "actual_dropout": getattr(recurrent, "dropout", None),
            "input_size": data.input_size, "n_classes": data.n_classes,
            "train_size": len(data.x_train), "val_size": len(data.x_val), "test_size": len(data.x_test),
            "class_names": " | ".join(data.class_names),
            "val_loss": result["val_metrics"]["loss"], "val_accuracy": result["val_metrics"]["accuracy"],
            "test_loss": None if result["metrics"] is None else result["metrics"]["loss"],
            "test_accuracy": None if result["metrics"] is None else result["metrics"]["accuracy"],
            "test_evaluated": result["metrics"] is not None,
            "representation": result.get("representation", "sequence"),
            "readout": result.get("readout", "model.forward"),
            "parameter_count": result["parameter_count"], "elapsed_seconds": result["elapsed_seconds"],
            "model_structure": str(result["model"]),
        })
        records.append(record)
    return records


def export_experiments_csv(results: list[dict[str, object]], path: str | Path) -> Path:
    """保存 UTF-8 BOM 的实验表，便于 Excel 查看中文；不会触发测试集评估。"""

    records = experiment_records(results)
    if not records:
        raise ValueError("请先完成至少一组实验再导出 CSV。")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return destination


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
    *,
    evaluate_test: bool = True,
) -> list[dict[str, object]]:
    """按相同配置依次运行 RNN、LSTM 和 GRU。"""

    return [run_experiment(config, evaluate_test=evaluate_test) for config in comparison_configs(base, lengths)]
