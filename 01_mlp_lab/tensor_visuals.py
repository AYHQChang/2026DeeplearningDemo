"""Visual explanations used by ``01_认识数据与张量.ipynb``.

The notebook keeps each code cell short.  These helpers contain only the
Matplotlib layout needed to turn shapes, indexing and batches into pictures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import torch

from mlp_lab.data import DatasetBundle
from mlp_lab.plots import POINT_COLORS, configure_chinese_font


def _finish(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_frame_on(False)


def _draw_number_grid(
    axis: plt.Axes,
    values: np.ndarray,
    selected: set[tuple[int, int]] | None = None,
) -> None:
    rows, columns = values.shape
    selected = selected or set()
    axis.set_xlim(-0.5, columns - 0.5)
    axis.set_ylim(rows - 0.5, -0.5)
    for row in range(rows):
        for column in range(columns):
            active = (row, column) in selected
            axis.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor="#FBBF24" if active else "#DBEAFE",
                    edgecolor="#1E3A8A",
                    linewidth=2 if active else 1,
                )
            )
            axis.text(column, row, f"{values[row, column]:g}", ha="center", va="center", fontsize=12)
    axis.set_aspect("equal")
    _finish(axis)


def _draw_stack(axis: plt.Axes, groups: int, layers: int) -> None:
    colors = ("#BFDBFE", "#93C5FD", "#60A5FA")
    for group in range(groups):
        group_x = 0.08 + group * 0.47
        for layer in reversed(range(layers)):
            offset = layer * 0.045
            for row in range(2):
                for column in range(3):
                    axis.add_patch(
                        Rectangle(
                            (group_x + column * 0.09 + offset, 0.24 + row * 0.13 + offset),
                            0.085,
                            0.12,
                            facecolor=colors[layer % len(colors)],
                            edgecolor="#1E3A8A",
                            linewidth=0.9,
                        )
                    )
        if groups > 1:
            axis.text(group_x + 0.14, 0.12, f"样本 {group}", ha="center", fontsize=9)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal")
    _finish(axis)


def plot_tensor_dimensions() -> plt.Figure:
    """Show scalar, vector, matrix, channel stack and batch side by side."""

    configure_chinese_font()
    figure, axes = plt.subplots(1, 5, figsize=(16, 3.3), constrained_layout=True)

    axes[0].text(
        0.5,
        0.5,
        "7",
        ha="center",
        va="center",
        fontsize=28,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#DBEAFE", "edgecolor": "#1E3A8A"},
    )
    axes[0].set_title("0 维：标量")
    axes[0].set_xlabel("shape = ( )")
    _finish(axes[0])

    _draw_number_grid(axes[1], np.array([[1, 2, 3, 4]]))
    axes[1].set_title("1 维：向量")
    axes[1].set_xlabel("shape = (4,)")

    _draw_number_grid(axes[2], np.arange(1, 7).reshape(2, 3))
    axes[2].set_title("2 维：矩阵")
    axes[2].set_xlabel("shape = (2, 3)")

    _draw_stack(axes[3], groups=1, layers=2)
    axes[3].set_title("3 维：多张矩阵")
    axes[3].set_xlabel("shape = (2, 2, 3)")

    _draw_stack(axes[4], groups=2, layers=2)
    axes[4].set_title("4 维：一批样本")
    axes[4].set_xlabel("shape = (2, 2, 2, 3)")

    figure.suptitle("维度表示定位一个数需要多少个索引", fontsize=15, fontweight="bold")
    return figure


def plot_tensor_indexing(tensor: torch.Tensor) -> plt.Figure:
    """Highlight common two-dimensional indexing operations."""

    if tensor.ndim != 2:
        raise ValueError("索引示意图需要一个二维 Tensor。")
    configure_chinese_font()
    values = tensor.detach().cpu().numpy()
    rows, columns = values.shape
    selections = (
        ("完整 Tensor：a", {(r, c) for r in range(rows) for c in range(columns)}),
        ("第一行：a[0]", {(0, c) for c in range(columns)}),
        ("第二列：a[:, 1]", {(r, 1) for r in range(rows)}),
        ("一个数：a[1, 2]", {(1, 2)}),
    )
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)
    for axis, (title, selected) in zip(axes, selections):
        _draw_number_grid(axis, values, selected)
        axis.set_title(title)
    figure.suptitle("索引就是沿着各个轴定位数据", fontsize=14, fontweight="bold")
    return figure


def plot_sample_to_batch(sample: torch.Tensor, batch: torch.Tensor) -> plt.Figure:
    """Connect one feature vector to a mini-batch of feature vectors."""

    if sample.shape != (2,) or batch.ndim != 2 or batch.shape[1] != 2:
        raise ValueError("示例要求 sample.shape == (2,) 且 batch.shape == (B, 2)。")
    configure_chinese_font()
    sample_values = sample.detach().cpu().numpy()
    batch_values = batch.detach().cpu().numpy()
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    _draw_number_grid(axes[0], sample_values.reshape(1, 2), {(0, 0), (0, 1)})
    axes[0].set_title("一个样本")
    axes[0].set_xlabel("[x1, x2]，shape = (2,)")

    axes[1].scatter(batch_values[:, 0], batch_values[:, 1], s=70, color="#2563EB")
    axes[1].scatter(sample_values[0], sample_values[1], s=230, facecolors="none", edgecolors="#F59E0B", linewidths=3)
    for index, point in enumerate(batch_values):
        axes[1].annotate(f"样本 {index}", point, xytext=(5, 6), textcoords="offset points", fontsize=9)
    axes[1].set_title("每一行对应一个点")
    axes[1].set_xlabel("特征 x1")
    axes[1].set_ylabel("特征 x2")
    axes[1].grid(alpha=0.2)

    _draw_number_grid(axes[2], batch_values, {(0, 0), (0, 1)})
    axes[2].set_title("把样本按行叠起来")
    axes[2].set_xlabel(f"batch.shape = {tuple(batch.shape)}")
    figure.suptitle("从一个样本到一个 mini-batch", fontsize=14, fontweight="bold")
    return figure


def plot_image_tensor_shapes(image: torch.Tensor, batch_size: int = 4) -> plt.Figure:
    """Show how a 28x28 image gains channel and batch axes."""

    if image.ndim != 2 or tuple(image.shape) != (28, 28):
        raise ValueError("图像示例需要 shape == (28, 28)。")
    configure_chinese_font()
    image_np = image.detach().cpu().numpy()
    image_with_channel = image.unsqueeze(0)
    batch = image_with_channel.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.5), constrained_layout=True)

    axes[0].imshow(image_np, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("单张灰度图")
    axes[0].set_xlabel("[H, W] = [28, 28]")

    axes[1].imshow(image_np, cmap="Blues", vmin=0, vmax=1)
    for spine in axes[1].spines.values():
        spine.set_color("#F59E0B")
        spine.set_linewidth(4)
    axes[1].set_title("增加通道轴 C")
    axes[1].set_xlabel("[C, H, W] = [1, 28, 28]")

    strip = np.concatenate([image_np] * batch_size, axis=1)
    axes[2].imshow(strip, cmap="Blues", vmin=0, vmax=1)
    for boundary in range(1, batch_size):
        axes[2].axvline(boundary * 28 - 0.5, color="white", linewidth=3)
    axes[2].set_title("增加批次轴 B")
    axes[2].set_xlabel(f"[B, C, H, W] = [{batch_size}, 1, 28, 28]")

    flattened = batch.reshape(batch_size, -1).numpy()
    axes[3].imshow(flattened, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[3].set_title("MLP 把每张图拉直")
    axes[3].set_xlabel(f"[B, H×W] = [{batch_size}, 784]")
    axes[3].set_ylabel("样本编号")

    for axis in axes[:3]:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("同一张图像，不同模型需要不同的 Tensor 形状", fontsize=14, fontweight="bold")
    return figure


def plot_moons_tensor_map(data: DatasetBundle, rows: int = 6) -> plt.Figure:
    """Link points in the moons dataset to rows in X and entries in y."""

    configure_chinese_font()
    x = data.x_train.detach().cpu().numpy()
    y = data.y_train.detach().cpu().numpy()
    shown = min(rows, len(x))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for class_id in range(data.n_classes):
        mask = y == class_id
        axes[0].scatter(x[mask, 0], x[mask, 1], s=24, alpha=0.75, color=POINT_COLORS[class_id], label=f"类别 {class_id}")
    axes[0].scatter(x[0, 0], x[0, 1], s=260, facecolors="none", edgecolors="#F59E0B", linewidths=3)
    axes[0].annotate("X[0]", x[0], xytext=(10, 10), textcoords="offset points", fontweight="bold")
    axes[0].set_title("图上的一个点")
    axes[0].set_xlabel("标准化特征 x1")
    axes[0].set_ylabel("标准化特征 x2")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].axis("off")
    table_rows = [[str(index), f"{x[index, 0]:.3f}", f"{x[index, 1]:.3f}", str(y[index])] for index in range(shown)]
    table = axes[1].table(cellText=table_rows, colLabels=["索引", "x1", "x2", "y"], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.55)
    for column in range(4):
        table[(1, column)].set_facecolor("#FDE68A")
        table[(0, column)].set_facecolor("#DBEAFE")
    axes[1].set_title("Tensor 中的一行（前 6 个样本）")

    axes[2].axis("off")
    cards = (
        (0.82, "全部输入 X_train", f"shape = {tuple(data.x_train.shape)}\n每行 2 个特征", "#DBEAFE"),
        (0.57, "全部标签 y_train", f"shape = {tuple(data.y_train.shape)}\n每个样本 1 个类别", "#FECACA"),
        (0.32, "一个输入 X_train[0]", f"shape = {tuple(data.x_train[0].shape)}", "#FEF3C7"),
        (0.10, "一个标签 y_train[0]", "shape = ( )，它是一个标量", "#DCFCE7"),
    )
    for y_position, title, detail, color in cards:
        axes[2].text(
            0.5,
            y_position,
            f"{title}\n{detail}",
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "#475569"},
        )
    axes[2].set_title("shape 告诉我们数据如何组织")
    figure.suptitle("同一份 moons 数据：点、表格与 Tensor 是三种视角", fontsize=14, fontweight="bold")
    return figure


def plot_mlp_shape_flow(batch_size: int, hidden_size: int, n_classes: int) -> plt.Figure:
    """Draw the shape journey through a two-layer MLP."""

    configure_chinese_font()
    figure, axis = plt.subplots(figsize=(12, 3.2), constrained_layout=True)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 3)
    _finish(axis)
    boxes = (
        (0.3, "输入 X", f"[{batch_size}, 2]", "#DBEAFE"),
        (3.3, "Linear + ReLU", f"[{batch_size}, {hidden_size}]", "#DCFCE7"),
        (6.6, "Linear", f"[{batch_size}, {n_classes}]", "#FEF3C7"),
        (9.6, "argmax", f"[{batch_size}]", "#FECACA"),
    )
    for left, title, shape, color in boxes:
        axis.add_patch(FancyBboxPatch((left, 0.85), 2.1, 1.3, boxstyle="round,pad=0.08", facecolor=color, edgecolor="#334155", linewidth=1.5))
        axis.text(left + 1.05, 1.67, title, ha="center", va="center", fontsize=11, fontweight="bold")
        axis.text(left + 1.05, 1.25, shape, ha="center", va="center", fontsize=11)
    for start, end in ((2.4, 3.3), (5.4, 6.6), (8.7, 9.6)):
        axis.add_patch(FancyArrowPatch((start, 1.5), (end, 1.5), arrowstyle="-|>", mutation_scale=16, color="#475569"))
    axis.text(6, 0.35, "批次大小 B 始终保留；变化的是每个样本的表示维度", ha="center", fontsize=11)
    axis.set_title("MLP 训练时真正流动的是一批 Tensor", fontsize=14, fontweight="bold")
    return figure
