"""CNN Notebook 使用的图片、卷积机制和训练诊断图。"""

from functools import lru_cache
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
from torch.nn import functional as F

from .data import ImageDatasetBundle
from .engine import TrainingResult, shifted_accuracy


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


def _image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze().numpy()


def plot_digit_gallery(data: ImageDatasetBundle) -> plt.Figure:
    """每个类别展示一张图片，先建立任务直觉。"""

    configure_chinese_font()
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.8), constrained_layout=True)
    for label, axis in enumerate(axes.ravel()):
        index = int(torch.where(data.y_train == label)[0][0])
        axis.imshow(_image(data.x_train[index]), cmap="gray_r", vmin=0, vmax=1)
        axis.set_title(f"标签 {label}")
        axis.axis("off")
    fig.suptitle(f"离线手写数字：{len(data.x_train) + len(data.x_test)} 张 {data.image_size}×{data.image_size} 灰度图", fontsize=15)
    return fig


def plot_pixel_and_shape(data: ImageDatasetBundle, index: int = 0) -> plt.Figure:
    """把普通图片、像素矩阵与 `[B,C,H,W]` 对齐。"""

    configure_chinese_font()
    image = _image(data.x_train[index])
    label = int(data.y_train[index])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    axes[0].imshow(image, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title(f"人眼看到的数字｜标签 {label}")
    axes[0].axis("off")
    axes[1].imshow(image, cmap="Blues", vmin=0, vmax=1)
    if data.image_size <= 10:
        for row in range(data.image_size):
            for column in range(data.image_size):
                axes[1].text(column, row, f"{image[row, column]:.1f}", ha="center", va="center", fontsize=7)
    axes[1].set_title("模型接收的像素数值（已缩放到 0–1）")
    axes[1].set_xlabel("宽度 W")
    axes[1].set_ylabel("高度 H")
    axes[2].axis("off")
    axes[2].text(
        0.05, 0.92,
        "单张图片\n"
        f"[H, W] = [{data.image_size}, {data.image_size}]\n\n"
        "加入通道\n"
        f"[C, H, W] = [{data.channels}, {data.image_size}, {data.image_size}]\n\n"
        "组成 Batch\n"
        f"[B, C, H, W] = [64, {data.channels}, {data.image_size}, {data.image_size}]\n\n"
        "以后换成 28×28 时，只改变 H、W。",
        va="top", fontsize=13, linespacing=1.45,
    )
    axes[2].set_title("Tensor shape：每一维代表什么？")
    fig.suptitle("同一张图片：视觉、数值与 Tensor shape", fontsize=15)
    return fig


def _teaching_kernels() -> dict[str, torch.Tensor]:
    return {
        "垂直边缘": torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32),
        "水平边缘": torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32),
        "锐化": torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32),
    }


def plot_convolution_demo(data: ImageDatasetBundle, index: int = 0) -> plt.Figure:
    """显示人工 3×3 卷积核及其输出响应。"""

    configure_chinese_font()
    sample = data.x_train[index:index + 1]
    kernels = _teaching_kernels()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    axes[0, 0].imshow(_image(sample), cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title("输入图片 X")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.55, "局部窗口逐元素相乘\n再求和，得到一个输出位置", ha="center", va="center", fontsize=12)
    for column, (name, kernel) in enumerate(kernels.items(), start=1):
        response = F.conv2d(sample, kernel.view(1, 1, 3, 3), padding=1)
        axes[0, column].imshow(kernel.numpy(), cmap="coolwarm", vmin=-5, vmax=5)
        for row in range(3):
            for kernel_column in range(3):
                axes[0, column].text(kernel_column, row, f"{kernel[row, kernel_column]:g}", ha="center", va="center", fontsize=11)
        axes[0, column].set_title(f"3×3 卷积核｜{name}")
        axes[1, column].imshow(_image(response), cmap="coolwarm")
        axes[1, column].set_title(f"输出 Feature map｜{name}")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    fig.suptitle("同一张图片经过不同卷积核，会突出不同局部模式", fontsize=16)
    return fig


def plot_pooling_demo(data: ImageDatasetBundle, index: int = 0) -> plt.Figure:
    """比较 Max Pooling 与 Average Pooling 对空间尺寸和响应的影响。"""

    configure_chinese_font()
    sample = data.x_train[index:index + 1]
    kernel = _teaching_kernels()["垂直边缘"].view(1, 1, 3, 3)
    feature = F.relu(F.conv2d(sample, kernel, padding=1))
    max_pooled = F.max_pool2d(feature, 2)
    avg_pooled = F.avg_pool2d(feature, 2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, tensor, title in zip(
        axes,
        (feature, max_pooled, avg_pooled),
        (f"卷积响应｜{data.image_size}×{data.image_size}", f"Max Pooling｜{data.image_size // 2}×{data.image_size // 2}", f"Average Pooling｜{data.image_size // 2}×{data.image_size // 2}"),
    ):
        axis.imshow(_image(tensor), cmap="magma", vmin=0)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("Pooling 不学习卷积核：它只汇总局部区域", fontsize=15)
    return fig


def _predictions(result: TrainingResult) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(result.model.parameters()).device
    result.model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(result.model(result.data.x_test.to(device)), dim=1).cpu()
    return probabilities.argmax(dim=1), probabilities


def _plot_sample_strip(axis: plt.Axes, images: torch.Tensor, labels: list[str], title: str) -> None:
    if len(images) == 0:
        axis.text(0.5, 0.5, "本次没有符合条件的样本", ha="center", va="center")
        axis.axis("off")
        return
    strip = np.concatenate([_image(image) for image in images], axis=1)
    axis.imshow(strip, cmap="gray_r", vmin=0, vmax=1)
    width = images.shape[-1]
    for boundary in range(1, len(images)):
        axis.axvline(boundary * width - 0.5, color="#2563EB", linewidth=1)
    axis.set_title(title + "\n" + "｜".join(labels), fontsize=10)
    axis.axis("off")


def plot_training_overview(result: TrainingResult) -> plt.Figure:
    """用训练曲线、混淆矩阵和样本证据回答模型学得怎样。"""

    configure_chinese_font()
    predictions, probabilities = _predictions(result)
    truth = result.data.y_test
    matrix = torch.zeros(result.data.n_classes, result.data.n_classes, dtype=torch.int64)
    for expected, predicted in zip(truth, predictions):
        matrix[int(expected), int(predicted)] += 1
    correct = torch.where(predictions == truth)[0][:5]
    wrong = torch.where(predictions != truth)[0][:5]
    fig, axes = plt.subplot_mosaic(
        [["loss", "accuracy", "confusion"], ["correct", "errors", "info"]],
        figsize=(16, 8.5), constrained_layout=True,
    )
    epochs = np.arange(1, result.config.epochs + 1)
    axes["loss"].plot(epochs, result.train_loss, color="#7C3AED")
    axes["loss"].set_title("Loss：优化过程是否稳定？")
    axes["loss"].set_xlabel("Epoch")
    axes["loss"].set_ylabel("Cross-Entropy")
    axes["accuracy"].plot(epochs, result.train_accuracy, color="#2563EB", label="训练")
    axes["accuracy"].plot(epochs, result.test_accuracy, color="#DC2626", linestyle="--", label="测试")
    axes["accuracy"].set_ylim(0, 1.02)
    axes["accuracy"].set_title("Accuracy：训练与测试差距")
    axes["accuracy"].set_xlabel("Epoch")
    axes["accuracy"].legend()
    axes["confusion"].imshow(matrix.numpy(), cmap="Blues")
    axes["confusion"].set_title("混淆矩阵：哪些类别容易混淆？")
    axes["confusion"].set_xlabel("预测标签")
    axes["confusion"].set_ylabel("真实标签")
    axes["confusion"].set_xticks(range(result.data.n_classes))
    axes["confusion"].set_yticks(range(result.data.n_classes))
    _plot_sample_strip(
        axes["correct"], result.data.x_test[correct],
        [f"真{int(truth[i])}/预测{int(predictions[i])}" for i in correct], "正确样本",
    )
    _plot_sample_strip(
        axes["errors"], result.data.x_test[wrong],
        [f"真{int(truth[i])}/预测{int(predictions[i])}" for i in wrong], "错误样本",
    )
    sample_index = int(wrong[0]) if len(wrong) else 0
    top_values, top_indices = probabilities[sample_index].topk(3)
    axes["info"].axis("off")
    axes["info"].text(
        0.03, 0.95,
        f"实验卡\n配置：{result.config.name}\n"
        f"参数量：{result.parameter_count:,}\n训练耗时：{result.elapsed_seconds:.2f} 秒\n"
        f"测试准确率：{result.final_test_accuracy:.1%}\n右移 1 像素：{shifted_accuracy(result):.1%}\n\n"
        f"示例真实标签：{int(truth[sample_index])}\n"
        + "Top-3：\n"
        + "\n".join(f"数字 {int(label)}：{float(value):.1%}" for value, label in zip(top_values, top_indices)),
        va="top", fontsize=12, linespacing=1.35,
    )
    for key in ("loss", "accuracy"):
        axes[key].grid(alpha=0.2)
    fig.suptitle(f"训练诊断：{result.config.name}", fontsize=17)
    return fig


def plot_feature_maps(result: TrainingResult, sample_index: int = 0, max_channels: int = 8) -> plt.Figure:
    """展示同一图片在两层卷积后的多个通道响应。"""

    configure_chinese_font()
    device = next(result.model.parameters()).device
    sample = result.data.x_test[sample_index:sample_index + 1].to(device)
    result.model.eval()
    with torch.no_grad():
        maps = result.model.feature_maps(sample)
        prediction = int(result.model(sample).argmax(dim=1).item())
    first = maps["第一层卷积"][0].cpu()
    second = maps["第二层卷积"][0].cpu()
    columns = max_channels + 1
    fig, axes = plt.subplots(2, columns, figsize=(2.0 * columns, 4.8), constrained_layout=True)
    axes[0, 0].imshow(_image(sample), cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title(f"输入\n真{int(result.data.y_test[sample_index])}/预测{prediction}")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "每一列是一个通道\n不是一张新的原图", ha="center", va="center", fontsize=10)
    for column in range(max_channels):
        for row, tensor, name in ((0, first, "Conv1"), (1, second, "Conv2")):
            axis = axes[row, column + 1]
            if column < len(tensor):
                axis.imshow(_image(tensor[column]), cmap="magma")
                axis.set_title(f"{name}\n通道 {column}")
            axis.axis("off")
    fig.suptitle("Feature maps：同一层的不同卷积核产生不同通道响应", fontsize=16)
    return fig


def plot_pooling_comparison(results: list[TrainingResult]) -> plt.Figure:
    """同时比较准确率、平移测试、参数量和训练曲线。"""

    if not results:
        raise ValueError("results 不能为空。")
    configure_chinese_font()
    labels = [result.config.name for result in results]
    clean = [result.final_test_accuracy for result in results]
    shifted = [shifted_accuracy(result) for result in results]
    parameters = [result.parameter_count for result in results]
    times = [result.elapsed_seconds for result in results]
    positions = np.arange(len(results))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    width = 0.35
    axes[0, 0].bar(positions - width / 2, clean, width, label="原始测试集", color="#2563EB")
    axes[0, 0].bar(positions + width / 2, shifted, width, label="右移 1 像素", color="#F97316")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_xticks(positions, labels)
    axes[0, 0].set_title("准确率与轻微平移：不能只看一个数字")
    axes[0, 0].legend()
    axes[0, 1].bar(labels, parameters, color="#7C3AED")
    axes[0, 1].set_title("参数量：不池化会保留更大的空间特征")
    axes[0, 1].set_ylabel("可训练参数")
    for position, value in enumerate(parameters):
        axes[0, 1].text(position, value, f"{value:,}", ha="center", va="bottom")
    for result in results:
        axes[1, 0].plot(range(1, result.config.epochs + 1), result.train_loss, label=result.config.name)
        axes[1, 1].plot(range(1, result.config.epochs + 1), result.test_accuracy, label=result.config.name)
    axes[1, 0].set_title("训练 Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Cross-Entropy")
    axes[1, 1].set_title("测试 Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[0, 1].text(0.98, 0.04, "训练时间：" + "｜".join(f"{name} {value:.2f}s" for name, value in zip(labels, times)), transform=axes[0, 1].transAxes, ha="right", fontsize=9)
    for axis in axes.ravel():
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Pooling 单变量对比：空间压缩、参数量与表现的共同变化", fontsize=16)
    return fig


def format_result_table(results: list[TrainingResult]) -> str:
    lines = ["CNN 配置对比", "-" * 78, f"{'配置':<22}{'测试准确率':>12}{'右移准确率':>12}{'参数量':>12}{'耗时(秒)':>12}", "-" * 78]
    for result in results:
        lines.append(f"{result.config.name:<22}{result.final_test_accuracy:>11.1%}{shifted_accuracy(result):>11.1%}{result.parameter_count:>12,}{result.elapsed_seconds:>12.2f}")
    lines.append("-" * 78)
    return "\n".join(lines)


def print_result_table(results: list[TrainingResult]) -> None:
    print("\n" + format_result_table(results))
