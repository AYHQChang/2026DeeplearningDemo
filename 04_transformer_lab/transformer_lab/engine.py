"""注意力寻宝实验的安全配置、训练、评估和受控对比。"""

from dataclasses import dataclass, replace
from math import ceil
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    RetrievalDatasetBundle,
    make_dataloaders,
    make_retrieval_data,
    resolve_device,
    seed_everything,
)
from .model import TinyTransformerClassifier, estimate_parameter_count


@dataclass(frozen=True)
class ExperimentConfig:
    """一轮受控实验的全部可修改参数。"""

    name: str = "基线：有位置编码"
    content_length: int = 12
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 1
    dim_feedforward: int = 64
    dropout: float = 0.1
    activation: str = "gelu"
    use_positional_encoding: bool = True
    learning_rate: float = 0.003
    batch_size: int = 64
    epochs: int = 12
    train_size: int = 1200
    val_size: int = 300
    test_size: int = 300
    gradient_clip: float | None = 1.0
    seed: int = 42
    device: str = "cpu"


def update_count(config: ExperimentConfig) -> int:
    return ceil(config.train_size / config.batch_size) * config.epochs


def attention_budget(config: ExperimentConfig) -> int:
    return config.train_size * config.epochs * config.num_layers * (config.content_length + 1) ** 2


def validate_config(config: ExperimentConfig) -> None:
    """在生成数据和分配模型前拦截不适合共享服务器的配置。"""

    resolve_device(config.device)
    positive = (
        config.d_model, config.num_heads, config.num_layers,
        config.dim_feedforward, config.learning_rate, config.batch_size,
        config.epochs, config.train_size, config.val_size, config.test_size,
    )
    if min(positive) <= 0:
        raise ValueError("模型维度、学习率、批量、轮数和数据量都必须大于 0。")
    if config.content_length < 3 or config.content_length % 3 != 0:
        raise ValueError("content_length 至少为 3，且必须能被 3 整除。")
    if config.content_length > 24:
        raise ValueError("content_length 不能超过 24。")
    if config.train_size > 3000 or max(config.val_size, config.test_size) > 1000:
        raise ValueError("train_size 不能超过 3000，val_size 和 test_size 不能超过 1000。")
    if config.epochs > 30 or config.batch_size > 128:
        raise ValueError("epochs 不能超过 30，batch_size 不能超过 128。")
    if config.d_model > 64 or config.num_heads > 4 or config.num_layers > 2:
        raise ValueError("d_model≤64、num_heads≤4、num_layers≤2。")
    if config.dim_feedforward > 128:
        raise ValueError("dim_feedforward 不能超过 128。")
    if config.d_model % config.num_heads != 0:
        raise ValueError("d_model 必须能被 num_heads 整除。")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout 必须位于 [0, 1) 区间。")
    if config.gradient_clip is not None and config.gradient_clip <= 0:
        raise ValueError("gradient_clip 必须大于 0，或设为 None。")
    if update_count(config) > 1500:
        raise ValueError("预计参数更新次数不能超过 1500。")
    if attention_budget(config) > 12_000_000:
        raise ValueError("Attention 组合预算不能超过 12000000。")
    if estimate_parameter_count(config.d_model, config.num_layers, config.dim_feedforward) > 150_000:
        raise ValueError("模型参数量不能超过 150000。")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def evaluate_model(
    model: TinyTransformerClassifier,
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, object]:
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


def train_model(
    model: TinyTransformerClassifier,
    loaders: dict[str, DataLoader],
    config: ExperimentConfig,
) -> tuple[dict[str, list[float]], float]:
    device = resolve_device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "grad_norm": []}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(config.epochs):
        model.train()
        total_loss = total_examples = total_grad_norm = batches = 0.0
        for batch_x, batch_y in loaders["train"]:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            max_norm = config.gradient_clip if config.gradient_clip is not None else float("inf")
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            total_examples += len(batch_x)
            total_grad_norm += float(grad_norm)
            batches += 1
        val_metrics = evaluate_model(model, loaders["val"], config.device)
        history["train_loss"].append(total_loss / total_examples)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["grad_norm"].append(total_grad_norm / batches)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return history, time.perf_counter() - start


def predict_proba(
    model: TinyTransformerClassifier,
    token_ids: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    resolved = resolve_device(device)
    model.to(resolved).eval()
    with torch.no_grad():
        return model(token_ids.to(resolved)).softmax(dim=1).cpu()


def attention_for_sample(
    model: TinyTransformerClassifier,
    token_ids: torch.Tensor,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """只返回所选样本最后一层的概率和 Attention，避免缓存整批权重。"""

    resolved = resolve_device(device)
    model.to(resolved).eval()
    with torch.no_grad():
        logits, weights = model.forward_with_attention(token_ids[:1].to(resolved))
    return logits.softmax(dim=1).cpu(), weights.cpu()


def run_experiment(
    config: ExperimentConfig,
    data: RetrievalDatasetBundle | None = None,
) -> dict[str, object]:
    validate_config(config)
    seed_everything(config.seed)
    data = data or make_retrieval_data(
        content_length=config.content_length,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed,
    )
    if data.content_length != config.content_length:
        raise ValueError("传入数据的 content_length 与配置不一致。")
    actual_sizes = (len(data.x_train), len(data.x_val), len(data.x_test))
    expected_sizes = (config.train_size, config.val_size, config.test_size)
    if actual_sizes != expected_sizes:
        raise ValueError(f"传入数据量 {actual_sizes} 与配置 {expected_sizes} 不一致。")
    loaders = make_dataloaders(data, batch_size=config.batch_size, seed=config.seed)
    model = TinyTransformerClassifier(
        vocab_size=data.vocab_size,
        n_classes=len(data.class_names),
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation=config.activation,
        use_positional_encoding=config.use_positional_encoding,
        max_length=25,
    )
    parameter_count = count_parameters(model)
    if parameter_count > 150_000:
        raise ValueError("模型参数量不能超过 150000。")
    history, elapsed_seconds = train_model(model, loaders, config)
    metrics = evaluate_model(model, loaders["test"], config.device)
    return {
        "config": config, "data": data, "loaders": loaders, "model": model,
        "history": history, "metrics": metrics,
        "elapsed_seconds": elapsed_seconds,
        "parameter_count": parameter_count,
        "update_count": update_count(config),
        "attention_budget": attention_budget(config),
    }


def comparison_configs(base: ExperimentConfig) -> list[ExperimentConfig]:
    """生成位置编码主对照与 1/2/4 Head 次对照。"""

    return [
        replace(base, name="有位置编码｜4 Heads", use_positional_encoding=True, num_heads=4),
        replace(base, name="无位置编码｜4 Heads", use_positional_encoding=False, num_heads=4),
        replace(base, name="有位置编码｜1 Head", use_positional_encoding=True, num_heads=1),
        replace(base, name="有位置编码｜2 Heads", use_positional_encoding=True, num_heads=2),
    ]


def compare_experiments(base: ExperimentConfig) -> list[dict[str, object]]:
    data = make_retrieval_data(
        content_length=base.content_length, train_size=base.train_size,
        val_size=base.val_size, test_size=base.test_size, seed=base.seed,
    )
    return [run_experiment(config, data=data) for config in comparison_configs(base)]


def evaluate_length_transfer(
    result: dict[str, object],
    content_length: int = 24,
    test_size: int = 300,
) -> dict[str, object]:
    """不重新训练，直接检查模型在更长平衡序列上的表现。"""

    config = result["config"]
    if not isinstance(config, ExperimentConfig):
        raise TypeError("result 中缺少 ExperimentConfig。")
    if content_length < 3 or content_length % 3 != 0 or content_length > 24:
        raise ValueError("迁移长度必须位于 3 到 24 且能被 3 整除。")
    data = make_retrieval_data(
        content_length=content_length, train_size=3, val_size=3,
        test_size=test_size, seed=config.seed + 10_000,
    )
    loader = make_dataloaders(data, batch_size=config.batch_size, seed=config.seed)["test"]
    return evaluate_model(result["model"], loader, config.device)
