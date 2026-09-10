"""Transformer Notebook 使用的 token、位置编码、Attention 与训练图。"""

from functools import lru_cache
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
import numpy as np
import torch

from .data import CLASS_NAMES, TOKEN_NAMES, RetrievalDatasetBundle, resolve_device
from .model import SinusoidalPositionalEncoding, TinyTransformerClassifier


BUNDLED_CJK_FONT = (
    Path(__file__).resolve().parents[2]
    / "01_mlp_lab" / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
)
SYSTEM_CJK_FONTS = (
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC",
    "WenQuanYi Zen Hei", "Arial Unicode MS",
)
TOKEN_COLORS = ("#ef5350", "#43a047", "#4285f4", "#ffd166")


@lru_cache(maxsize=1)
def _preferred_chinese_font() -> str:
    if BUNDLED_CJK_FONT.is_file():
        font_manager.fontManager.addfont(str(BUNDLED_CJK_FONT))
        return font_manager.FontProperties(fname=BUNDLED_CJK_FONT).get_name()
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in SYSTEM_CJK_FONTS:
        if candidate in available:
            return candidate
    warnings.warn("未找到中文字体，Matplotlib 中文可能显示为方框。", RuntimeWarning)
    return "DejaVu Sans"


def configure_chinese_font() -> str:
    selected = _preferred_chinese_font()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def plot_token_sequence(
    token_ids: torch.Tensor,
    labels: torch.Tensor,
    index: int = 0,
) -> plt.Figure:
    """把一条寻宝序列画成带位置编号的彩色 token。"""

    configure_chinese_font()
    row = token_ids[index].detach().cpu().tolist()
    label = int(labels[index])
    fig, ax = plt.subplots(figsize=(13, 2.6), constrained_layout=True)
    for position, token_id in enumerate(row):
        rect = patches.FancyBboxPatch(
            (position - 0.43, 0.1), 0.86, 0.72,
            boxstyle="round,pad=0.02", facecolor=TOKEN_COLORS[token_id],
            edgecolor="#333333",
        )
        ax.add_patch(rect)
        ax.text(position, 0.46, TOKEN_NAMES[token_id], ha="center", va="center")
        ax.text(position, -0.02, str(position), ha="center", va="top", fontsize=9)
    ax.set_xlim(-0.6, len(row) - 0.4)
    ax.set_ylim(-0.2, 1.05)
    ax.axis("off")
    ax.set_title(f"标签：找回第 0 位的{CLASS_NAMES[label]}｜每种颜色总数相同")
    return fig


def plot_tensor_structure(data: RetrievalDatasetBundle) -> plt.Figure:
    configure_chinese_font()
    sample = data.x_train[:9].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    image = axes[0].imshow(sample, aspect="auto", cmap="viridis", vmin=0, vmax=3)
    axes[0].set_xlabel("Time / token 位置")
    axes[0].set_ylabel("Batch 中的样本")
    axes[0].set_title("token id 张量 [B,T]")
    fig.colorbar(image, ax=axes[0], ticks=range(4), label="token id")
    axes[1].axis("off")
    lines = [
        f"完整训练输入：{tuple(data.x_train.shape)}",
        "Embedding 后：[B, T, d_model]",
        "Attention 权重：[B, Head, T, T]",
        "QUERY 分类输出：[B, 3]",
    ]
    axes[1].text(0.05, 0.85, "\n\n".join(lines), va="top", fontsize=12)
    axes[1].set_title("Shape 流程")
    return fig


def plot_positional_encoding(d_model: int = 32, max_length: int = 25) -> plt.Figure:
    configure_chinese_font()
    values = SinusoidalPositionalEncoding(d_model, max_length).encoding[0].numpy().T
    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("位置 position")
    ax.set_ylabel("Embedding 维度")
    ax.set_title("正弦位置编码：相同 token 在不同位置得到不同位置信号")
    fig.colorbar(image, ax=ax)
    return fig


def plot_attention_weights(
    weights: torch.Tensor,
    labels: list[str] | tuple[str, ...] | None = None,
    title: str = "Attention 权重",
) -> plt.Figure:
    configure_chinese_font()
    values = weights.detach().cpu().numpy()
    names = labels or [str(index) for index in range(len(values))]
    fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    ax.bar(range(len(values)), values, color="#5b8ff9")
    ax.set_xticks(range(len(values)), names, rotation=45, ha="right")
    ax.set_ylim(0, max(1.0, float(values.max()) * 1.15))
    ax.set_ylabel("Softmax 权重")
    ax.set_title(title)
    return fig


def plot_model_attention(
    model: TinyTransformerClassifier,
    token_ids: torch.Tensor,
    device: str = "cpu",
) -> plt.Figure:
    """只绘制一条样本最后一层的平均 Attention 和 QUERY 行。"""

    configure_chinese_font()
    resolved = resolve_device(device)
    model.to(resolved).eval()
    sample = token_ids[:1].to(resolved)
    with torch.no_grad():
        _, weights = model.forward_with_attention(sample)
    matrix = weights[0].mean(dim=0).cpu().numpy()
    ids = sample[0].cpu().tolist()
    names = [f"{i}:{TOKEN_NAMES[token]}" for i, token in enumerate(ids)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    image = axes[0].imshow(matrix, cmap="magma", vmin=0, vmax=max(0.2, matrix.max()))
    axes[0].set_xlabel("Key 位置")
    axes[0].set_ylabel("Query 位置")
    axes[0].set_title("最后一层各 Head 的平均 Attention")
    fig.colorbar(image, ax=axes[0])
    axes[1].bar(range(len(ids)), matrix[-1], color=[TOKEN_COLORS[token] for token in ids])
    axes[1].set_xticks(range(len(ids)), names, rotation=55, ha="right", fontsize=8)
    axes[1].set_ylim(0, max(0.4, float(matrix[-1].max()) * 1.15))
    axes[1].set_title("末尾 QUERY 对各位置的权重")
    axes[1].set_ylabel("Attention")
    return fig


def plot_training_history(history: dict[str, list[float]], title: str = "训练过程") -> plt.Figure:
    configure_chinese_font()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_accuracy"], color="#2a9d8f")
    axes[1].axhline(1 / 3, color="#777777", linestyle="--", label="随机猜测")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[2].plot(epochs, history["grad_norm"], color="#e76f51")
    axes[2].set_title("裁剪前 Gradient norm")
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.suptitle(title)
    return fig


def plot_confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    title: str = "测试集混淆矩阵",
) -> plt.Figure:
    configure_chinese_font()
    matrix = np.zeros((3, 3), dtype=int)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        matrix[target, prediction] += 1
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            ax.text(column, row, matrix[row, column], ha="center", va="center")
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("模型预测")
    ax.set_ylabel("真实标签")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    return fig


def print_result_table(results: list[dict[str, object]]) -> None:
    print(f"{'实验':<24} {'测试Acc':>8} {'参数量':>9} {'更新':>6} {'组合预算':>11} {'秒':>7}")
    for result in results:
        config = result["config"]
        print(
            f"{config.name:<24} {result['metrics']['accuracy']:>8.1%} "
            f"{result['parameter_count']:>9,} {result['update_count']:>6} "
            f"{result['attention_budget']:>11,} {result['elapsed_seconds']:>7.2f}"
        )


def plot_comparison(results: list[dict[str, object]]) -> plt.Figure:
    configure_chinese_font()
    labels = [result["config"].name for result in results]
    accuracies = [result["metrics"]["accuracy"] for result in results]
    parameters = [result["parameter_count"] for result in results]
    budgets = [result["attention_budget"] for result in results]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].bar(labels, accuracies)
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title("测试 Accuracy")
    axes[1].bar(labels, parameters, color="#2a9d8f")
    axes[1].set_title("参数量")
    axes[2].bar(labels, budgets, color="#f4a261")
    axes[2].set_title("Attention 组合预算")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    return fig
