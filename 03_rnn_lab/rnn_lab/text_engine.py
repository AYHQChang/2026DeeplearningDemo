"""文本任务轻量训练适配：探索默认只评价验证集。"""

import csv
from copy import deepcopy
import math
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

import torch
from torch import nn

from .data import resolve_device, seed_everything
from .engine import count_parameters
from .observation import evaluating
from .text_data import (TextDataBundle, build_vocab, encode_text, make_text_dataloaders,
                        save_text_preprocessing, text_data_diagnostics, validate_vocab)
from .text_model import TextClassifier


@dataclass(frozen=True)
class TextExperimentConfig:
    name: str = "字符级 GRU"
    cell_type: str = "gru"
    embedding_dim: int = 32
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    readout: str = "last"
    max_length: int = 100
    batch_size: int = 64
    epochs: int = 6
    learning_rate: float = 0.003
    optimizer: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    selection: str = "last"  # 旧教师参考保持末轮；学生入口显式使用 best_val_accuracy。
    gradient_clip: float | None = 1.0
    seed: int = 42
    device: str = "cpu"


def _validate_config(config):
    integer_fields = ("embedding_dim", "hidden_size", "num_layers", "max_length", "batch_size", "epochs")
    if any(isinstance(getattr(config, key), bool) or not isinstance(getattr(config, key), int)
           for key in integer_fields):
        raise ValueError("维度、长度、批量和轮数必须为正整数。")
    if not all(math.isfinite(value) for value in (config.learning_rate, config.momentum, config.weight_decay, config.dropout)):
        raise ValueError("训练参数必须为有限数。")
    if config.optimizer not in {"adam", "sgd", "adamw"}:
        raise ValueError("optimizer 为 adam、sgd 或 adamw。")
    if not 0 <= config.momentum < 1 or config.weight_decay < 0:
        raise ValueError("momentum 应在 [0,1)，weight_decay 不能为负。")
    if config.selection not in {"last", "best_val_accuracy"}:
        raise ValueError("selection 为 last 或 best_val_accuracy。")
    if config.gradient_clip is not None and not math.isfinite(config.gradient_clip):
        raise ValueError("gradient_clip 必须为有限正数或 None。")
    if config.device in {"auto", "cuda"}:
        raise ValueError("请使用 cpu，或课程分配的 cuda:编号。")
    if min(config.embedding_dim, config.hidden_size, config.num_layers, config.max_length,
           config.batch_size, config.epochs, config.learning_rate) <= 0:
        raise ValueError("维度、长度、批量、轮数和学习率必须为正。")
    if config.cell_type not in {"rnn", "lstm", "gru", "mean_embedding"} or config.readout not in {"last", "mean"}:
        raise ValueError("循环单元或读出配置不正确。")
    if not 0 <= config.dropout < 1 or (config.gradient_clip is not None and config.gradient_clip <= 0):
        raise ValueError("dropout 需位于 [0,1)，gradient_clip 为正数或 None。")


def _model_device(model):
    return next(model.parameters()).device


def evaluate_text_model(model: nn.Module, loader) -> dict:
    """不搬动模型，不更改其模式和已有梯度；返回与样本对应的原文和 ID。"""
    device = _model_device(model)
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    loss_total, targets, probabilities, texts, ids, lengths = 0.0, [], [], [], [], []
    with evaluating(model), torch.no_grad():
        for batch in loader:
            logits = model(batch["token_ids"].to(device), batch["lengths"].to(device))
            loss_total += float(loss_function(logits, batch["labels"].to(device)))
            targets.append(batch["labels"])
            probabilities.append(logits.softmax(dim=1).cpu())
            texts.extend(batch["texts"])
            ids.extend(batch["ids"])
            lengths.append(batch["lengths"])
    if not targets:
        raise ValueError("评估划分不能为空。")
    y = torch.cat(targets)
    probs = torch.cat(probabilities)
    preds = probs.argmax(dim=1)
    f1s = []
    for label in range(probs.shape[1]):
        tp = int(((preds == label) & (y == label)).sum())
        denominator = int((preds == label).sum()) + int((y == label).sum())
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return {"loss": loss_total / len(y), "accuracy": float((preds == y).float().mean()),
            "macro_f1": sum(f1s) / len(f1s), "targets": y, "predictions": preds,
            "probabilities": probs, "texts": texts, "ids": ids, "lengths": torch.cat(lengths)}


def run_text_experiment(config: TextExperimentConfig, data: TextDataBundle, vocab=None,
                        model: nn.Module | None = None, *, evaluate_test: bool = False) -> dict:
    """默认 CPU 单线程；固定训练轮数，不读取测试成绩选择设置。"""
    _validate_config(config)
    seed_everything(config.seed)
    if vocab is None:
        vocab = build_vocab(row["text"] for row in data.splits["train"])
    validate_vocab(vocab)
    vocab = dict(vocab)
    loaders = make_text_dataloaders(data, vocab, config.max_length, config.batch_size, config.seed)
    if model is None:
        model = TextClassifier(len(vocab), data.n_classes, config.embedding_dim, config.hidden_size,
                               config.cell_type, config.num_layers, config.dropout, config.readout)
    device = resolve_device(config.device)
    model.to(device)
    sample = next(iter(loaders["val"]))
    with evaluating(model), torch.no_grad():
        output = model(sample["token_ids"].to(device), sample["lengths"].to(device))
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != (len(sample["labels"]), data.n_classes):
        raise ValueError("文本模型应返回 [B,K] logits，K 由数据类别配置确定。")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("文本模型 logits 含非有限值。")
    optimizer_class = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}[config.optimizer]
    optimizer_options = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.optimizer == "sgd":
        optimizer_options["momentum"] = config.momentum
    optimizer = optimizer_class(model.parameters(), **optimizer_options)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": [], "grad_norm": []}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    best_key, best_state, best_val, selected_epoch = None, None, None, config.epochs
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, count, grad_sum, batches = 0.0, 0, 0.0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["token_ids"].to(device), batch["lengths"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("训练 Loss 不是有限值，请检查学生实现。")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip or float("inf"))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch["labels"])
            count += len(batch["labels"])
            grad_sum += float(norm)
            batches += 1
        val = evaluate_text_model(model, loaders["val"])
        history["train_loss"].append(loss_sum / count)
        history["grad_norm"].append(grad_sum / batches)
        for name in ("loss", "accuracy", "macro_f1"):
            history[f"val_{name}"].append(val[name])
        # 验证 Accuracy 最大；同分选 Loss 更低，再同分保留更早轮次。不读取测试。
        key = (val["accuracy"], -val["loss"])
        if config.selection == "best_val_accuracy" and (best_key is None or key > best_key):
            best_key, best_state, best_val = key, deepcopy(model.state_dict()), val
            selected_epoch = epoch
    if config.selection == "best_val_accuracy":
        model.load_state_dict(best_state)
        val = best_val
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {"config": config, "data": data, "vocab": vocab, "model": model, "loaders": loaders,
              "history": history, "val_metrics": val, "metrics": None,
              "selected_epoch": selected_epoch, "optimizer_steps": config.epochs * len(loaders["train"]),
              "elapsed_seconds": time.perf_counter() - started, "parameter_count": count_parameters(model),
              "diagnostics": text_data_diagnostics(data, vocab, config.max_length)}
    if evaluate_test:
        evaluate_text_final(result)
    return result


def model_fingerprint(model: nn.Module) -> str:
    """包含参数和 buffer 的实际字节；测试缓存不依赖对象 ID 或训练轮数。"""
    digest = hashlib.sha256()
    digest.update((type(model).__module__ + "." + type(model).__qualname__ + str(model)).encode("utf-8"))
    digest.update(str((getattr(model, "readout", None), getattr(model, "cell_type", None))).encode("utf-8"))
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _check_cached_test(result):
    if result["metrics"] is not None and (result.get("test_model_fingerprint") != model_fingerprint(result["model"])
            or result.get("test_encoding_fingerprint") != _encoding_fingerprint(result)):
        raise RuntimeError("最终测试后模型参数已改变；旧测试缓存不能代表当前模型。请创建新的实验记录。")


def _encoding_fingerprint(result):
    payload = {"vocab": result["vocab"], "max_length": result["config"].max_length,
               "class_names": result["data"].class_names}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def evaluate_text_final(result: dict) -> dict:
    """选定方案后显式测试。模型状态变化时拒绝复用首次测试缓存。"""
    _check_cached_test(result)
    if result["metrics"] is None:
        result["metrics"] = evaluate_text_model(result["model"], result["loaders"]["test"])
        result["test_model_fingerprint"] = model_fingerprint(result["model"])
        result["test_encoding_fingerprint"] = _encoding_fingerprint(result)
    return result["metrics"]


def predict_text(result: dict, text: str) -> dict:
    encoded = encode_text(text, result["vocab"], result["config"].max_length)
    model = result["model"]
    device = _model_device(model)
    ids = torch.tensor([encoded["token_ids"]], dtype=torch.int64, device=device)
    lengths = torch.tensor([encoded["length"]], dtype=torch.int64, device=device)
    with evaluating(model), torch.no_grad():
        probabilities = model(ids, lengths).softmax(dim=1)[0].cpu()
    label = int(probabilities.argmax())
    return {**encoded, "text": text, "prediction": label,
            "predicted_class": result["data"].class_names[label],
            "probabilities": {name: float(probabilities[index]) for index, name in enumerate(result["data"].class_names)}}


def text_experiment_records(results: list[dict]) -> list[dict]:
    records = []
    for result in results:
        _check_cached_test(result)
        data, config, model = result["data"], result["config"], result["model"]
        recurrent = getattr(model, "recurrent", None)
        record = {"task": "chinese_sentiment_classification", **asdict(config),
            "data_version": data.version, "training_split": "train", "validation_split": "val",
            "test_split": "test", "representation": "characters", "vocab_size": len(result["vocab"]),
            "model_class": type(model).__name__, "actual_recurrent_type": type(recurrent).__name__ if recurrent is not None else "none",
            "implementation": result.get("implementation", "pytorch"),
            "actual_hidden_size": getattr(recurrent, "hidden_size", None),
            "actual_num_layers": getattr(recurrent, "num_layers", None),
            "effective_readout": "masked_embedding_mean" if config.cell_type == "mean_embedding" else getattr(model, "readout", config.readout),
            "actual_recurrent_dropout": getattr(recurrent, "dropout", None),
            "device": str(_model_device(model)), "cpu_threads": torch.get_num_threads(), "torch_version": torch.__version__,
            "n_classes": data.n_classes, "class_names": " | ".join(data.class_names),
            "train_size": len(data.splits["train"]), "val_size": len(data.splits["val"]), "test_size": len(data.splits["test"]),
            "parameter_count": result["parameter_count"], "elapsed_seconds": result["elapsed_seconds"],
            "selected_epoch": result.get("selected_epoch", config.epochs),
            "optimizer_steps": result.get("optimizer_steps", config.epochs * len(result["loaders"]["train"])),
            "course_level": result.get("course_level", ""),
            "level_coefficient": {1: .8, 2: .9, 3: 1.0}.get(result.get("course_level"), ""),
            "val_loss": result["val_metrics"]["loss"], "val_accuracy": result["val_metrics"]["accuracy"],
            "val_macro_f1": result["val_metrics"]["macro_f1"], "test_evaluated": result["metrics"] is not None,
            "test_model_fingerprint": result.get("test_model_fingerprint", "")}
        record["provisional_score"] = (100 * result["metrics"]["accuracy"] * record["level_coefficient"]
            if result["metrics"] is not None and record["level_coefficient"] != "" else None)
        for key in ("loss", "accuracy", "macro_f1"):
            record[f"test_{key}"] = None if result["metrics"] is None else result["metrics"][key]
        for split, diagnostics in result["diagnostics"].items():
            for key in ("truncated_fraction", "unknown_fraction"):
                record[f"{split}_{key}"] = diagnostics[key]
        records.append(record)
    return records


def export_text_experiments_csv(results: list[dict], path: str | Path) -> Path:
    records = text_experiment_records(results)
    if not records:
        raise ValueError("至少需要一组实验。")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return destination


def save_text_run(result: dict, output_root: str | Path) -> Path:
    """每次保存生成新目录，不覆盖往次结果；保存后仍可在其中添加图。"""
    _check_cached_test(result)
    run_dir = Path(output_root) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    save_text_preprocessing(result["vocab"], result["config"].max_length, result["data"].class_names,
                            run_dir / "preprocessing.json")
    (run_dir / "config.json").write_text(json.dumps(asdict(result["config"]), ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(result["model"].state_dict(), run_dir / "model.pt")
    export_text_experiments_csv([result], run_dir / "experiments.csv")
    metrics = {"validation": {key: result["val_metrics"][key] for key in ("loss", "accuracy", "macro_f1")},
               "test": None if result["metrics"] is None else {key: result["metrics"][key] for key in ("loss", "accuracy", "macro_f1")},
               "model_fingerprint": model_fingerprint(result["model"]), "diagnostics": result["diagnostics"],
               "selected_epoch": result.get("selected_epoch", result["config"].epochs),
               "selection": result["config"].selection}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["epoch", *result["history"]]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(result["config"].epochs):
            writer.writerow({"epoch": epoch + 1, **{key: values[epoch] for key, values in result["history"].items()}})
    result["run_dir"] = run_dir
    return run_dir
