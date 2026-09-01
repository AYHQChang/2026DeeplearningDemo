"""MLP 实验开始前的数据与任务说明入口。

这个文件在训练模型之前回答五个问题：

1. 月牙、同心圆和螺旋数据分别长什么样；
2. 训练集与测试集如何按 75%/25% 分层划分；
3. 为什么标准化参数只能用训练集估计，以及标准化前后发生了什么；
4. MLP 接收什么形状的输入、输出什么形状的 logits、最终完成什么任务；
5. 默认基线包含哪些组件，接下来如何运行并只修改一个变量。

脚本生成两张互补图：``01_data_explanation.png`` 展示数据几何、划分、标准化
和类别数量；``02_task_flow.png`` 展示从原始数据到训练证据的完整任务流程。数据
全部复用 ``core.make_dataset``，不会复制另一套生成或划分逻辑，也不会启动训练。

常用命令：

    python data_and_task.py
    python data_and_task.py --dataset spiral --save-dir outputs --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

from core import (
    DatasetBundle,
    POINT_COLORS,
    configure_chinese_font,
    make_dataset,
    save_or_show,
)


DATASET_NAMES = {
    "moons": "月牙",
    "circles": "同心圆",
    "spiral": "三分类螺旋",
}
DEFAULT_NOISE = {"moons": 0.20, "circles": 0.20, "spiral": 0.45}


def build_parser() -> argparse.ArgumentParser:
    """创建数据与任务说明脚本的命令行解析器。

    Returns:
        支持数据选择、样本量、噪声、随机种子和图像输出设置的解析器。
        ``all`` 会把三类数据放在同一张图的三行中，便于直接比较。
    """

    parser = argparse.ArgumentParser(description="MLP 实验前置说明：数据、任务与基线")
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASET_NAMES),
        default="all",
        help="展示全部数据或只展示一种数据",
    )
    parser.add_argument("--n-samples", type=int, default=600, help="每种数据的样本总数")
    parser.add_argument("--noise", type=float, default=None, help="可选：覆盖数据默认噪声")
    parser.add_argument("--seed", type=int, default=42, help="数据生成和划分的随机种子")
    parser.add_argument("--save-dir", type=Path, default=None, help="可选：保存两张说明图的目录")
    parser.add_argument("--no-show", action="store_true", help="不弹出图形窗口")
    return parser


def _scatter_classes(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    marker: str = "o",
    alpha: float = 0.78,
) -> None:
    """按类别使用固定颜色绘制二维散点。

    Args:
        axis: 接收图形的 Matplotlib 坐标轴。
        x: ``float [n_samples, 2]`` 二维坐标。
        y: ``int [n_samples]`` 类别编号。
        n_classes: 数据中的类别总数。
        marker: Matplotlib 点形状；圆点通常表示训练样本，叉号表示测试样本。
        alpha: 点的不透明度。

    Returns:
        ``None``。颜色始终与训练图一致，避免在不同图中重新学习图例。
    """

    for class_id in range(n_classes):
        mask = y == class_id
        axis.scatter(
            x[mask, 0],
            x[mask, 1],
            s=20 if marker == "o" else 30,
            marker=marker,
            color=POINT_COLORS[class_id],
            alpha=alpha,
            linewidths=1.2 if marker == "x" else 0.0,
        )


def print_dataset_summary(bundle: DatasetBundle) -> None:
    """在终端输出一种数据的形状、划分、类别和标准化参数。

    Args:
        bundle: ``make_dataset`` 返回的数据对象。

    Returns:
        ``None``。输出中的均值和尺度均来自训练集；测试集从未参与拟合
        ``StandardScaler``，因此不会产生数据泄漏。
    """

    train_counts = np.bincount(bundle.y_train.numpy(), minlength=bundle.n_classes)
    test_counts = np.bincount(bundle.y_test.numpy(), minlength=bundle.n_classes)
    print(f"\n{DATASET_NAMES[bundle.dataset_name]}（{bundle.dataset_name}）")
    print(f"  输入形状：train={tuple(bundle.x_train.shape)}，test={tuple(bundle.x_test.shape)}")
    print(f"  标签形状：train={tuple(bundle.y_train.shape)}，test={tuple(bundle.y_test.shape)}")
    print(f"  类别数量：train={train_counts.tolist()}，test={test_counts.tolist()}")
    print(f"  训练集原始均值：{np.round(bundle.feature_mean, 4).tolist()}")
    print(f"  训练集原始尺度：{np.round(bundle.feature_scale, 4).tolist()}")
    print(f"  标准化后训练均值：{np.round(bundle.x_train.numpy().mean(axis=0), 4).tolist()}")
    print(f"  标准化后训练标准差：{np.round(bundle.x_train.numpy().std(axis=0), 4).tolist()}")


def plot_data_explanation(bundles: list[DatasetBundle]) -> plt.Figure:
    """绘制原始几何、数据划分、标准化和类别分布四列对照图。

    Args:
        bundles: 一个或多个 ``DatasetBundle``；每种数据占据一行。

    Returns:
        Matplotlib ``Figure``。每行坐标和颜色保持一致，四列依次回答“长什么
        样、怎样划分、标准化改变什么、类别是否均衡”。

    Raises:
        ValueError: ``bundles`` 为空。
    """

    if not bundles:
        raise ValueError("bundles 不能为空。")
    configure_chinese_font()
    figure, axes = plt.subplots(
        len(bundles),
        4,
        figsize=(18, 4.6 * len(bundles)),
        squeeze=False,
        constrained_layout=True,
    )
    split_legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6B7280", label="训练集", markersize=7),
        Line2D([0], [0], marker="x", color="#6B7280", label="测试集", markersize=7, linestyle="none"),
    ]

    for row, bundle in enumerate(bundles):
        raw_x = np.vstack((bundle.x_train_raw, bundle.x_test_raw))
        raw_y = np.concatenate((bundle.y_train.numpy(), bundle.y_test.numpy()))
        scaled_x = np.vstack((bundle.x_train.numpy(), bundle.x_test.numpy()))
        name = DATASET_NAMES[bundle.dataset_name]

        _scatter_classes(axes[row, 0], raw_x, raw_y, bundle.n_classes)
        axes[row, 0].set_title(f"{name}：原始几何")

        _scatter_classes(
            axes[row, 1], bundle.x_train_raw, bundle.y_train.numpy(), bundle.n_classes, marker="o", alpha=0.62
        )
        _scatter_classes(
            axes[row, 1], bundle.x_test_raw, bundle.y_test.numpy(), bundle.n_classes, marker="x", alpha=0.95
        )
        axes[row, 1].set_title("分层划分：训练 75% / 测试 25%")
        axes[row, 1].legend(handles=split_legend, loc="best")

        _scatter_classes(axes[row, 2], scaled_x, raw_y, bundle.n_classes)
        axes[row, 2].axhline(0, color="#6B7280", linewidth=0.8, alpha=0.6)
        axes[row, 2].axvline(0, color="#6B7280", linewidth=0.8, alpha=0.6)
        axes[row, 2].set_title("标准化后：训练均值约为 0")

        class_ids = np.arange(bundle.n_classes)
        train_counts = np.bincount(bundle.y_train.numpy(), minlength=bundle.n_classes)
        test_counts = np.bincount(bundle.y_test.numpy(), minlength=bundle.n_classes)
        axes[row, 3].bar(class_ids - 0.18, train_counts, width=0.36, color="#2563EB", label="训练集")
        axes[row, 3].bar(class_ids + 0.18, test_counts, width=0.36, color="#F97316", label="测试集")
        axes[row, 3].set_xticks(class_ids, [f"类别 {value}" for value in class_ids])
        axes[row, 3].set_ylabel("样本数")
        axes[row, 3].set_title("类别数量：分层划分保持比例")
        axes[row, 3].legend()
        axes[row, 3].grid(axis="y", alpha=0.2)

        for column in range(3):
            axes[row, column].set_xlabel("特征 x1")
            axes[row, column].set_ylabel("特征 x2")
            axes[row, column].set_aspect("equal", adjustable="datalim")
            axes[row, column].grid(alpha=0.15)

    figure.suptitle("数据说明：几何形状、固定划分、标准化与类别比例", fontsize=18)
    return figure


def plot_task_flow(dataset_name: str, n_classes: int) -> plt.Figure:
    """绘制从二维数据到可比较训练证据的 MLP 任务流程。

    Args:
        dataset_name: 当前示例数据的英文名称。
        n_classes: 输出类别数，决定最后一层 logits 的宽度。

    Returns:
        一个横向流程 ``Figure``。方框给出每个阶段的张量形状或关键约束，底部
        给出默认基线和后续只修改一个组件的实验顺序。
    """

    configure_chinese_font()
    figure, axis = plt.subplots(figsize=(17, 5.4), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    stages = [
        ("离线二维数据", f"{DATASET_NAMES[dataset_name]}\nX: [N, 2]  y: [N]"),
        ("固定数据划分", "训练 75%\n测试 25%\nstratify=y"),
        ("训练集标准化", "只 fit 训练集\n再 transform 两组"),
        ("MLP 基线", "2 -> 32 -> 32\nReLU + Dropout 0"),
        ("分类目标", f"logits: [N, {n_classes}]\nCross-Entropy"),
        ("训练证据", "边界 + loss\naccuracy + gradient norm"),
    ]
    centers = np.linspace(0.075, 0.925, len(stages))
    width, height = 0.12, 0.30
    for index, ((title, detail), center) in enumerate(zip(stages, centers)):
        box = FancyBboxPatch(
            (center - width / 2, 0.50),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#EFF6FF" if index < 3 else "#F0FDF4",
            edgecolor="#2563EB" if index < 3 else "#16A34A",
            linewidth=1.6,
        )
        axis.add_patch(box)
        axis.text(center, 0.72, title, ha="center", va="center", fontsize=12, weight="bold")
        axis.text(center, 0.59, detail, ha="center", va="center", fontsize=10.5, linespacing=1.35)
        if index < len(stages) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (center + width / 2 + 0.006, 0.65),
                    (centers[index + 1] - width / 2 - 0.006, 0.65),
                    arrowstyle="-|>",
                    mutation_scale=17,
                    color="#4B5563",
                    linewidth=1.8,
                )
            )

    axis.text(
        0.5,
        0.31,
        "默认基线：hidden=(32, 32)｜ReLU｜Cross-Entropy｜Adam｜lr=0.02｜Dropout=0",
        ha="center",
        va="center",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FEF3C7", "edgecolor": "#D97706"},
    )
    axis.text(
        0.5,
        0.16,
        "下一步：运行 baseline → 选择一个组件 → 只修改该组件 → 对照图片与实验日志",
        ha="center",
        va="center",
        fontsize=13,
        color="#7C2D12",
    )
    figure.suptitle("MLP 要完成什么任务？", fontsize=19)
    return figure


def main() -> None:
    """生成数据、打印任务摘要并显示或保存两张前置说明图。

    Returns:
        ``None``。这个入口只做数据说明，不训练模型，也不产生实验性能日志；
        训练日志从随后运行 ``demo.py`` 时开始记录。

    Raises:
        SystemExit: 样本数小于 12 或噪声为负时给出命令行错误。
    """

    parser = build_parser()
    args = parser.parse_args()
    if args.n_samples < 12:
        parser.error("n-samples 至少为 12。")
    if args.noise is not None and args.noise < 0:
        parser.error("noise 不能为负数。")

    selected = list(DATASET_NAMES) if args.dataset == "all" else [args.dataset]
    bundles = [
        make_dataset(
            name,
            n_samples=args.n_samples,
            noise=DEFAULT_NOISE[name] if args.noise is None else args.noise,
            seed=args.seed,
        )
        for name in selected
    ]

    print("MLP 前置说明：输入是二维坐标，目标是预测离散类别。")
    print("模型输出 raw logits；训练时使用损失函数，评价时用 argmax 得到类别。")
    for bundle in bundles:
        print_dataset_summary(bundle)
    print("\n推荐流程：")
    print("  1. 先阅读两张说明图，明确数据、输入输出和默认基线。")
    print("  2. 运行：python demo.py --experiment baseline")
    print("  3. 再运行：python demo.py --experiment activation")
    print("  4. 修改一个参数后重复运行，并在 logs/history.csv 中比较。")

    data_figure = plot_data_explanation(bundles)
    save_or_show(
        data_figure,
        "01_data_explanation.png",
        args.save_dir,
        show=not args.no_show,
    )
    # 展示全部数据时，任务流程与 demo.py 的默认螺旋基线保持一致。
    task_bundle = next(
        (bundle for bundle in bundles if bundle.dataset_name == "spiral"),
        bundles[0],
    )
    task_figure = plot_task_flow(task_bundle.dataset_name, task_bundle.n_classes)
    save_or_show(
        task_figure,
        "02_task_flow.png",
        args.save_dir,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
