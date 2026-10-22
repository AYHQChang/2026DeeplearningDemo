"""文本实验图：验证集、有效字符位置与明确的颜色含义。"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .plots import configure_chinese_font
from .text_data import encode_text
from .text_engine import _check_cached_test


def plot_text_history(result: dict):
    configure_chinese_font()
    history = result["history"]
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7), constrained_layout=True)
    axes[0].plot(epochs, history["train_loss"], "o-", label="训练 Loss")
    axes[0].plot(epochs, history["val_loss"], "o-", label="验证 Loss")
    axes[0].set_ylabel("交叉熵 Loss")
    axes[1].plot(epochs, history["val_accuracy"], "o-", label="验证 Accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], "s--", label="验证宏平均 F1")
    axes[1].set_ylim(0, 1)
    axes[2].plot(epochs, history["grad_norm"], "o-", label="裁剪前梯度范数均值")
    if result["config"].gradient_clip is not None:
        axes[2].axhline(result["config"].gradient_clip, color="gray", linestyle=":", label="逐批裁剪阈值")
    axes[2].set_ylabel("梯度范数（每轮批次均值）")
    for axis in axes:
        axis.axvline(result.get("selected_epoch", result["config"].epochs), color="gray", linestyle="--", alpha=.6, label="保存轮次")
    for axis in axes:
        axis.set_xlabel("训练轮数")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(result["config"].name + "｜探索仅使用验证集")
    return fig


def plot_text_confusion(result: dict, split: str = "val"):
    configure_chinese_font()
    if split not in {"val", "test"}:
        raise ValueError("混淆矩阵仅使用 val 或已显式评价的 test。")
    if split == "test":
        _check_cached_test(result)
    metrics = result["val_metrics"] if split == "val" else result["metrics"]
    if metrics is None:
        raise ValueError("测试集尚未评价；本绘图函数不会自动触发测试。")
    classes = result["data"].class_names
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, pred in zip(metrics["targets"], metrics["predictions"]):
        matrix[int(target), int(pred)] += 1
    fig, axis = plt.subplots(figsize=(5, 4.4), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(int(matrix.max()), 1))
    for row in range(len(classes)):
        for col in range(len(classes)):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center",
                      color="white" if matrix[row, col] > matrix.max() * .55 else "black")
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
             yticklabels=classes, xlabel="预测类别", ylabel="真实类别",
             title=f"{'验证集' if split == 'val' else '最终测试集'}混淆矩阵｜格内为样本数")
    fig.colorbar(image, ax=axis, label="样本数")
    return fig


def plot_text_states(observation: dict, color_limit: float | None = None):
    """h/c 使用同一以 0 为中心的色阶，PAD 不显示为字符。"""
    configure_chinese_font()
    states = [("顶层隐藏状态 h", observation["hidden_states"])]
    if observation["cell_states"] is not None:
        states.append(("顶层细胞状态 c", observation["cell_states"]))
    length = len(observation["characters"])
    bound = max(float(state.abs().max()) for _, state in states) if color_limit is None else float(color_limit)
    if not np.isfinite(bound) or bound < 0:
        raise ValueError("color_limit 应为有限非负数。")
    bound = max(bound, 1e-6)
    fig, axes = plt.subplots(len(states), 1, figsize=(min(22, max(9, length * .21)), 2.8 * len(states)),
                             squeeze=False, sharex=True, constrained_layout=True)
    tick_indices = sorted(set(range(0, length, max(1, (length + 23) // 24))) | {length - 1})
    ticks = [f"{index + 1}\n{observation['characters'][index] if observation['characters'][index] != ' ' else '[空格]'}"
             for index in tick_indices]
    for axis, (title, state) in zip(axes[:, 0], states):
        image = axis.imshow(state.numpy().T, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
        axis.set_ylabel("隐藏单元编号")
        axis.set_title(title)
        axis.set_xticks(tick_indices, ticks)
    axes[-1, 0].set_xlabel("有效字符位置（从 1 开始）；右侧 PAD 已排除")
    fig.colorbar(image, ax=axes[:, 0].tolist(), label="状态数值；红为正、蓝为负（h/c 共用色阶）")
    suffix = "｜已截断" if observation["truncated"] else ""
    fig.suptitle(f"{observation['cell_type']}｜{length} 个编码位置{suffix}｜逐步核对误差 {observation['replay_max_error']:.2g}")
    return fig


def plot_text_comparison(results: list[dict]):
    configure_chinese_font()
    if not results:
        raise ValueError("至少需要一组实验。")
    labels = [result["config"].name for result in results]
    fig, axes = plt.subplots(1, 3, figsize=(max(11, len(results) * 2.7), 4), constrained_layout=True)
    for axis, key, title in zip(axes, ("accuracy", "parameter_count", "elapsed_seconds"),
                                ("验证 Accuracy", "可训练参数量", "训练与每轮验证耗时 / 秒")):
        values = [result["val_metrics"][key] if key == "accuracy" else result[key] for result in results]
        bars = axis.bar(labels, values, color="#4a829c")
        axis.bar_label(bars, fmt="%.3f" if key == "accuracy" else "%.1f", padding=3)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=22)
        axis.set_ylim(0, 1 if key == "accuracy" else max(values) * 1.2)
    fig.suptitle("相同 H 不代表相同容量；平均 Embedding 基线不利用字符顺序")
    return fig


def text_examples(result: dict, correct: bool | None = None, limit: int = 5) -> list[dict]:
    """按原验证样本顺序展示，不按模型概率挑选漂亮例子。"""
    metrics = result["val_metrics"]
    examples = []
    for index, text in enumerate(metrics["texts"]):
        target, prediction = int(metrics["targets"][index]), int(metrics["predictions"][index])
        if correct is not None and (target == prediction) != correct:
            continue
        encoded = encode_text(text, result["vocab"], result["config"].max_length)
        examples.append({"id": metrics["ids"][index], "text": text,
            "true_class": result["data"].class_names[target], "predicted_class": result["data"].class_names[prediction],
            "correct": target == prediction, "length": encoded["length"], "truncated": encoded["truncated"],
            "unknown_count": encoded["unknown_count"], "probabilities": metrics["probabilities"][index].tolist()})
        if len(examples) >= limit:
            break
    return examples
