"""RNN Notebook 使用的序列、张量、训练和模型对比图。"""

from functools import lru_cache
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
import numpy as np
import torch

from .data import SequenceDatasetBundle, resolve_device
from .engine import input_gradient_by_time
from .model import RecurrentClassifier


BUNDLED_CJK_FONT = (
    Path(__file__).resolve().parents[2]
    / "01_mlp_lab"
    / "assets"
    / "fonts"
    / "NotoSansCJKsc-Regular.otf"
)
SYSTEM_CJK_FONTS = (
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC",
    "WenQuanYi Zen Hei", "Arial Unicode MS",
)
CLASS_COLORS = ("#ef5350", "#43a047", "#4285f4")
CELL_COLORS = {"rnn": "#8e7cc3", "lstm": "#2a9d8f", "gru": "#f4a261"}


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
    """优先加载仓库字体，保证 Linux 服务器上的中文图片可读。"""

    selected = _preferred_chinese_font()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def plot_sequence_sample(
    x: torch.Tensor,
    y: torch.Tensor,
    index: int = 0,
    class_names: list[str] | tuple[str, ...] = ("红色", "绿色", "蓝色"),
) -> plt.Figure:
    """把一条 ``[T,F]`` 序列画成时间线和特征热力图。"""

    configure_chinese_font()
    sample = x[index].detach().cpu()
    label = int(y[index])
    length = sample.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.8), constrained_layout=True)

    axes[0].set_xlim(-0.6, length - 0.4)
    axes[0].set_ylim(-0.2, 1.2)
    axes[0].axis("off")
    visible_steps = sorted(set([0, 1, 2, length // 2, length - 3, length - 2, length - 1]))
    for step in visible_steps:
        if step == 0:
            face, text = CLASS_COLORS[label], f"秘密\n{class_names[label]}"
        elif step == length - 1:
            face, text = "#ffd166", "查询\n?"
        else:
            face, text = "#d9e2ec", "噪声"
        axes[0].add_patch(patches.FancyBboxPatch(
            (step - 0.38, 0.18), 0.76, 0.62,
            boxstyle="round,pad=0.03", facecolor=face, edgecolor="#455a64"
        ))
        axes[0].text(step, 0.49, text, ha="center", va="center", fontsize=9)
        axes[0].text(step, 0.03, f"t={step}", ha="center", va="top", fontsize=8)
    for left, right in zip(visible_steps[:-1], visible_steps[1:]):
        axes[0].annotate("", xy=(right - 0.42, 0.49), xytext=(left + 0.42, 0.49),
                         arrowprops={"arrowstyle": "->", "color": "#607d8b"})
    axes[0].set_title("一条延迟回忆序列：答案只在第一个时间步出现")

    heat = axes[1].imshow(sample.T.numpy(), aspect="auto", cmap="RdYlBu_r", vmin=-1, vmax=1)
    axes[1].set_xlabel("时间步 Time")
    axes[1].set_ylabel("特征 Feature")
    axes[1].set_yticks(range(6), ["红", "绿", "蓝", "噪声1", "噪声2", "查询"])
    axes[1].set_title(f"Tensor 切片 [T,F] = [{length},6]｜标签：{class_names[label]}")
    fig.colorbar(heat, ax=axes[1], shrink=0.75, label="特征值")
    return fig


def plot_tensor_structure(data: SequenceDatasetBundle, index: int = 0) -> plt.Figure:
    """将 ``[Batch, Time, Feature]`` 三个轴与一条实际数据对齐。"""

    configure_chinese_font()
    sample = data.x_train[index]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    image = axes[0].imshow(sample.T.numpy(), aspect="auto", cmap="RdYlBu_r", vmin=-1, vmax=1)
    axes[0].set_xlabel(f"Time：{data.sequence_length} 个时间步")
    axes[0].set_ylabel(f"Feature：{data.input_size} 个通道")
    axes[0].set_yticks(range(6), ["红", "绿", "蓝", "噪声1", "噪声2", "查询"])
    axes[0].set_title("从一个 Batch 中取出一条序列")
    fig.colorbar(image, ax=axes[0], shrink=0.75)
    axes[1].axis("off")
    axes[1].text(
        0.05, 0.92,
        f"训练输入 X_train\n{list(data.x_train.shape)}\n\n"
        f"Batch：{len(data.x_train)} 条序列\n"
        f"Time：{data.sequence_length} 个时间步\n"
        f"Feature：{data.input_size} 个特征\n\n"
        f"标签 y_train\n{list(data.y_train.shape)}\n每条序列对应一个类别",
        va="top", fontsize=14, linespacing=1.4,
    )
    axes[1].set_title("输入和标签的 shape")
    fig.suptitle("RNN 的三维输入：[Batch, Time, Feature]", fontsize=16)
    return fig


def plot_recurrence_diagram(steps: int = 6) -> plt.Figure:
    """用展开图说明同一个 RNN 单元如何沿时间共享参数。"""

    configure_chinese_font()
    fig, axis = plt.subplots(figsize=(13, 3.6), constrained_layout=True)
    axis.set_xlim(-0.7, steps - 0.3)
    axis.set_ylim(-1.0, 1.2)
    axis.axis("off")
    for step in range(steps):
        axis.add_patch(patches.FancyBboxPatch(
            (step - 0.32, -0.1), 0.64, 0.55, boxstyle="round,pad=0.04",
            facecolor="#bde0fe", edgecolor="#1976d2"
        ))
        axis.text(step, 0.18, f"RNN\nt={step}", ha="center", va="center")
        axis.annotate(f"x{step}", xy=(step, -0.12), xytext=(step, -0.68), ha="center",
                      arrowprops={"arrowstyle": "->", "color": "#455a64"})
        axis.annotate(f"y{step}", xy=(step, 0.48), xytext=(step, 0.92), ha="center",
                      arrowprops={"arrowstyle": "->", "color": "#455a64"})
        if step:
            axis.annotate("", xy=(step - 0.35, 0.18), xytext=(step - 0.65, 0.18),
                          arrowprops={"arrowstyle": "->", "color": "#d1495b", "lw": 2})
            axis.text(step - 0.5, 0.31, f"h{step - 1}", ha="center", color="#b23a48")
    axis.set_title("把循环展开：相同参数重复使用，隐藏状态 h 沿时间传递")
    return fig


def plot_training_history(history: dict[str, list[float]], title: str = "训练过程") -> plt.Figure:
    """同时显示训练损失、验证损失、准确率和梯度范数。"""

    configure_chinese_font()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].plot(epochs, history["train_loss"], label="训练 Loss", color="#6c4cff")
    axes[0].plot(epochs, history["val_loss"], label="验证 Loss", color="#ef476f")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(epochs, history["val_accuracy"], color="#118ab2")
    axes[1].axhline(1 / 3, color="#777", linestyle="--", label="随机猜测 33.3%")
    axes[1].set_ylim(0, 1.03); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("验证 Accuracy"); axes[1].legend(); axes[1].grid(alpha=.25)
    axes[2].plot(epochs, history["grad_norm"], color="#f4a261")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("梯度范数"); axes[2].grid(alpha=.25)
    fig.suptitle(title, fontsize=15)
    return fig


def plot_confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    class_names: list[str] | tuple[str, ...] = ("红色", "绿色", "蓝色"),
    title: str = "混淆矩阵",
) -> plt.Figure:
    """绘制三分类混淆矩阵，不增加额外依赖。"""

    configure_chinese_font()
    n_classes = len(class_names)
    matrix = torch.bincount(
        targets.to(torch.int64) * n_classes + predictions.to(torch.int64),
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes).numpy()
    fig, axis = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(n_classes):
        for column in range(n_classes):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(n_classes), class_names)
    axis.set_yticks(range(n_classes), class_names)
    axis.set_xlabel("模型预测"); axis.set_ylabel("真实秘密"); axis.set_title(title)
    fig.colorbar(image, ax=axis, shrink=0.8)
    return fig


def plot_hidden_states(
    model: RecurrentClassifier,
    x: torch.Tensor,
    device: str = "cpu",
    title: str = "隐藏状态随时间变化",
) -> plt.Figure:
    """显示一条序列在各时间步的隐藏向量；它是描述，不是思维证明。"""

    configure_chinese_font()
    resolved = resolve_device(device)
    model.to(resolved).eval()
    with torch.no_grad():
        states = model.sequence_features(x[:1].to(resolved))[0].cpu().T.numpy()
    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    image = axis.imshow(states, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xlabel("时间步 Time"); axis.set_ylabel("隐藏维度 Hidden feature")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, shrink=0.8, label="隐藏状态值")
    return fig


def plot_input_gradients(
    model: RecurrentClassifier,
    x: torch.Tensor,
    target: torch.Tensor,
    device: str = "cpu",
    title: str = "输出对各时间步输入的梯度",
) -> plt.Figure:
    """用对数轴绘制输入梯度，帮助观察远距离信号是否变小。"""

    configure_chinese_font()
    gradients = input_gradient_by_time(model, x, target, device).clamp_min(1e-12).numpy()
    fig, axis = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    axis.plot(np.arange(len(gradients)), gradients, marker="o", markersize=3, color="#7b2cbf")
    axis.set_yscale("log")
    axis.set_xlabel("时间步 Time"); axis.set_ylabel("输入梯度范数（log）")
    axis.set_title(title); axis.grid(alpha=.25)
    return fig


def plot_comparison(results: list[dict[str, object]]) -> plt.Figure:
    """比较模型在不同序列长度下的准确率、时间和参数量。"""

    configure_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for cell_type in ("rnn", "lstm", "gru"):
        selected = [item for item in results if item["config"].cell_type == cell_type]
        selected.sort(key=lambda item: item["config"].sequence_length)
        lengths = [item["config"].sequence_length for item in selected]
        accuracies = [item["metrics"]["accuracy"] for item in selected]
        axes[0].plot(lengths, accuracies, marker="o", label=cell_type.upper(), color=CELL_COLORS[cell_type])
    axes[0].axhline(1 / 3, color="#777", linestyle="--", label="随机猜测")
    axes[0].set_ylim(0, 1.03); axes[0].set_xlabel("序列长度"); axes[0].set_ylabel("测试 Accuracy"); axes[0].legend(); axes[0].grid(alpha=.25)

    labels = [f"{item['config'].cell_type.upper()}\nT={item['config'].sequence_length}" for item in results]
    colors = [CELL_COLORS[item["config"].cell_type] for item in results]
    axes[1].bar(labels, [item["elapsed_seconds"] for item in results], color=colors)
    axes[1].set_ylabel("训练时间（秒）"); axes[1].tick_params(axis="x", labelsize=8)
    axes[2].bar(labels, [item["parameter_count"] for item in results], color=colors)
    axes[2].set_ylabel("可训练参数量"); axes[2].tick_params(axis="x", labelsize=8)
    fig.suptitle("同一任务、同一组超参数：循环单元与序列长度对比", fontsize=15)
    return fig


def print_result_table(results: list[dict[str, object]]) -> None:
    """打印适合课堂截图的紧凑结果表。"""

    print(f"{'模型':<8}{'长度':>6}{'准确率':>10}{'参数量':>10}{'时间/秒':>10}")
    print("-" * 44)
    for item in results:
        config = item["config"]
        print(
            f"{config.cell_type.upper():<8}{config.sequence_length:>6}"
            f"{item['metrics']['accuracy']:>10.1%}{item['parameter_count']:>10,}"
            f"{item['elapsed_seconds']:>10.2f}"
        )
