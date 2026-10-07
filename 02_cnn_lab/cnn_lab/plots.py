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
from .engine import TrainingResult, evaluation_data, shifted_accuracy


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
    values = tensor.detach().cpu()
    if values.ndim == 4 and len(values) == 1:
        values = values[0]
    if values.ndim == 3:
        if values.shape[0] == 1:
            values = values[0]
        elif values.shape[0] == 3:
            values = values.permute(1, 2, 0)
    return values.numpy()


def _class_name(data: ImageDatasetBundle, label: int) -> str:
    return str(data.class_names[int(label)])


def plot_digit_gallery(data: ImageDatasetBundle) -> plt.Figure:
    """每个类别展示一张图片；函数名保留以兼容数字实验。"""

    configure_chinese_font()
    columns = min(5, data.n_classes)
    rows = int(np.ceil(data.n_classes / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(2.2 * columns, 2.5 * rows),
        constrained_layout=True, squeeze=False,
    )
    for label, axis in enumerate(axes.ravel()):
        axis.axis("off")
        if label >= data.n_classes:
            continue
        indices = torch.where(data.y_train == label)[0]
        if len(indices):
            axis.imshow(_image(data.x_train[int(indices[0])]), cmap="gray_r", vmin=0, vmax=1)
        else:
            axis.text(0.5, 0.5, "训练集中没有样本", ha="center", va="center")
        axis.set_title(f"{_class_name(data, label)}｜标签 {label}")
    total = len(data.x_train) + len(data.x_test)
    if data.x_val is not None:
        total += len(data.x_val)
    fig.suptitle(
        f"{data.dataset_name}：{data.n_classes} 类，共 {total} 张 "
        f"{data.image_size}×{data.image_size} 图片", fontsize=15,
    )
    return fig


def plot_pixel_and_shape(data: ImageDatasetBundle, index: int = 0) -> plt.Figure:
    """把普通图片、像素矩阵与 `[B,C,H,W]` 对齐。"""

    configure_chinese_font()
    image = _image(data.x_train[index])
    label = int(data.y_train[index])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    axes[0].imshow(image, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title(f"人眼看到的图片｜{_class_name(data, label)}（标签 {label}）")
    axes[0].axis("off")
    axes[1].imshow(image, cmap="Blues", vmin=0, vmax=1)
    if data.image_size <= 10 and image.ndim == 2:
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
        "组成 Batch（以 B=64 为例）\n"
        f"[B, C, H, W] = [64, {data.channels}, {data.image_size}, {data.image_size}]\n\n"
        f"分类输出 [B, K] 中，K = {data.n_classes}。",
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


def _predictions(result: TrainingResult, split: str = "auto") -> tuple[torch.Tensor, torch.Tensor]:
    images, _, _ = evaluation_data(result, split=split)
    device = next(result.model.parameters()).device
    result.model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(result.model(images.to(device)), dim=1).cpu()
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
    axis.set_title(title, fontsize=10)
    centers = np.arange(len(images)) * width + (width - 1) / 2
    descriptions = [label.replace("真", "真实", 1).replace("/预测", "\n预测") for label in labels]
    axis.set_xticks(centers, descriptions, fontsize=8)
    axis.tick_params(axis="x", length=0, pad=5)
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_accuracy_curve(axis: plt.Axes, result: TrainingResult, *, split: str, label: str) -> None:
    curve = result.val_accuracy if split == "val" else result.test_accuracy
    if curve is None or not curve:
        return
    if len(curve) == 1 and result.config.epochs > 1:
        axis.scatter([result.config.epochs], curve, label=label)
    else:
        axis.plot(range(1, len(curve) + 1), curve, label=label)


def plot_training_overview(result: TrainingResult, split: str = "auto") -> plt.Figure:
    """探索时看验证集；最终测试后可查看测试集诊断。"""

    configure_chinese_font()
    images, truth, evaluation_label = evaluation_data(result, split=split)
    predictions, probabilities = _predictions(result, split=split)
    matrix = torch.zeros(result.data.n_classes, result.data.n_classes, dtype=torch.int64)
    for expected, predicted in zip(truth, predictions):
        matrix[int(expected), int(predicted)] += 1
    correct = torch.where(predictions == truth)[0][:5]
    wrong = torch.where(predictions != truth)[0][:5]
    fig, axes = plt.subplot_mosaic(
        [["loss", "accuracy", "confusion"], ["correct", "errors", "info"]],
        figsize=(16, 8.5), constrained_layout=True,
    )
    epochs = np.arange(1, len(result.train_loss) + 1)
    axes["loss"].plot(epochs, result.train_loss, color="#7C3AED")
    axes["loss"].set_title("Loss：优化过程是否稳定？")
    axes["loss"].set_xlabel("Epoch")
    axes["loss"].set_ylabel("Cross-Entropy")
    axes["accuracy"].plot(range(1, len(result.train_accuracy) + 1), result.train_accuracy, color="#2563EB", label="训练")
    curve_split = "val" if result.val_accuracy is not None else "test"
    curve_label = "验证" if curve_split == "val" else "测试"
    _plot_accuracy_curve(axes["accuracy"], result, split=curve_split, label=curve_label)
    if curve_split == "val" and result.test_accuracy:
        axes["accuracy"].scatter([result.config.epochs], [result.final_test_accuracy], color="#059669", label="最终测试")
    axes["accuracy"].set_ylim(0, 1.02)
    axes["accuracy"].set_title(f"Accuracy：训练与{curve_label}差距")
    axes["accuracy"].set_xlabel("Epoch")
    axes["accuracy"].legend()
    axes["confusion"].imshow(matrix.numpy(), cmap="Blues")
    axes["confusion"].set_title(f"{evaluation_label}混淆矩阵：哪些类别容易混淆？")
    axes["confusion"].set_xlabel("预测类别")
    axes["confusion"].set_ylabel("真实类别")
    positions = range(result.data.n_classes)
    axes["confusion"].set_xticks(positions, result.data.class_names, rotation=45, ha="right")
    axes["confusion"].set_yticks(positions, result.data.class_names)
    sample_labels = lambda indices: [
        f"真{_class_name(result.data, int(truth[i]))}/预测{_class_name(result.data, int(predictions[i]))}"
        for i in indices
    ]
    _plot_sample_strip(axes["correct"], images[correct], sample_labels(correct), f"{evaluation_label}正确样本")
    _plot_sample_strip(axes["errors"], images[wrong], sample_labels(wrong), f"{evaluation_label}错误样本")
    sample_index = int(wrong[0]) if len(wrong) else 0
    top_k = min(3, result.data.n_classes)
    top_values, top_indices = probabilities[sample_index].topk(top_k)
    accuracy = float((predictions == truth).float().mean())
    test_note = ""
    if not result.test_accuracy:
        test_note = "测试集尚未评估；确定模型后再执行最终测试。\n\n"
    axes["info"].axis("off")
    axes["info"].text(
        0.03, 0.95,
        f"实验卡\n配置：{result.config.name}\n"
        f"数据：{result.data.dataset_name}（{result.data.n_classes} 类）\n"
        f"参数量：{result.parameter_count:,}\n训练耗时：{result.elapsed_seconds:.2f} 秒\n"
        f"{evaluation_label}准确率：{accuracy:.1%}\n"
        f"{evaluation_label}右移 1 像素：{shifted_accuracy(result, split=split):.1%}\n\n"
        + test_note
        + f"示例真实类别：{_class_name(result.data, int(truth[sample_index]))}\n"
        + f"Top-{top_k}：\n"
        + "\n".join(f"{_class_name(result.data, int(label))}：{float(value):.1%}" for value, label in zip(top_values, top_indices)),
        va="top", fontsize=11, linespacing=1.35,
    )
    for key in ("loss", "accuracy"):
        axes[key].grid(alpha=0.2)
    fig.suptitle(f"训练诊断：{result.config.name}", fontsize=17)
    return fig


def plot_feature_maps(
    result: TrainingResult, sample_index: int = 0, max_channels: int = 8,
    split: str = "auto",
) -> plt.Figure:
    """按模型返回的顺序展示特征图，支持不同数量的卷积块。"""

    if max_channels <= 0:
        raise ValueError("max_channels 必须大于 0。")
    feature_maps = getattr(result.model, "feature_maps", None)
    if not callable(feature_maps):
        raise ValueError("这个自定义模型没有 feature_maps()；请使用 show_features=False，或在模型中返回各层特征图。")
    configure_chinese_font()
    images, truth, evaluation_label = evaluation_data(result, split=split)
    if not 0 <= sample_index < len(images):
        raise IndexError(f"sample_index 必须在 0 到 {len(images) - 1} 之间。")
    device = next(result.model.parameters()).device
    sample = images[sample_index:sample_index + 1].to(device)
    result.model.eval()
    with torch.no_grad():
        maps = feature_maps(sample)
        prediction = int(result.model(sample).argmax(dim=1).item())
    if not isinstance(maps, dict) or not maps:
        raise ValueError("feature_maps() 应返回非空字典：层名 -> [B, C, H, W] Tensor。")
    tensors = []
    for name, tensor in maps.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or len(tensor) != 1:
            raise ValueError(f"特征图 {name} 应为 [1, C, H, W] Tensor。")
        tensors.append((str(name), tensor[0].cpu()))
    rows = len(tensors)
    shown_channels = min(max_channels, max(len(tensor) for _, tensor in tensors))
    columns = shown_channels + 1
    fig, axes = plt.subplots(rows, columns, figsize=(2.0 * columns, 2.4 * rows), constrained_layout=True, squeeze=False)
    for row, (name, tensor) in enumerate(tensors):
        if row == 0:
            axes[row, 0].imshow(_image(sample), cmap="gray_r", vmin=0, vmax=1)
            axes[row, 0].set_title(
                f"{evaluation_label}输入\n真{_class_name(result.data, int(truth[sample_index]))}"
                f"/预测{_class_name(result.data, prediction)}", fontsize=10,
            )
        else:
            axes[row, 0].text(0.5, 0.5, f"{name}\nC×H×W\n{tuple(tensor.shape)}", ha="center", va="center", fontsize=10)
        axes[row, 0].axis("off")
        for channel in range(shown_channels):
            axis = axes[row, channel + 1]
            if channel < len(tensor):
                axis.imshow(_image(tensor[channel]), cmap="magma")
                axis.set_title(f"{name}\n通道 {channel}", fontsize=10)
            axis.axis("off")
    fig.suptitle("Feature maps：每一行是一层，每一列是一个通道响应", fontsize=16)
    return fig


def _comparison_split(results: list[TrainingResult]) -> str:
    if not results:
        raise ValueError("results 不能为空。")
    if all(result.val_accuracy is not None for result in results):
        return "val"
    if all(result.test_accuracy for result in results):
        return "test"
    raise ValueError("对比需要共同的验证集，或已评估的测试集；请为所有实验提供相同的数据划分。")


def _comparison_accuracy(result: TrainingResult, split: str) -> float:
    return result.final_validation_accuracy if split == "val" else result.final_test_accuracy


def plot_pooling_comparison(results: list[TrainingResult]) -> plt.Figure:
    """比较同一数据划分上的准确率、平移、参数量和训练曲线。"""

    split = _comparison_split(results)
    evaluation_label = "验证" if split == "val" else "测试"
    configure_chinese_font()
    labels = [result.config.name for result in results]
    clean = [_comparison_accuracy(result, split) for result in results]
    shifted = [shifted_accuracy(result, split=split) for result in results]
    parameters = [result.parameter_count for result in results]
    times = [result.elapsed_seconds for result in results]
    positions = np.arange(len(results))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    width = 0.35
    axes[0, 0].bar(positions - width / 2, clean, width, label=f"原始{evaluation_label}集", color="#2563EB")
    axes[0, 0].bar(positions + width / 2, shifted, width, label="右移 1 像素", color="#F97316")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_xticks(positions, labels)
    axes[0, 0].set_title("准确率与轻微平移：同时观察两项表现")
    axes[0, 0].legend()
    axes[0, 1].bar(labels, parameters, color="#7C3AED")
    axes[0, 1].set_title("参数量：空间尺寸、通道数和分类层共同影响")
    axes[0, 1].set_ylabel("可训练参数")
    for position, value in enumerate(parameters):
        axes[0, 1].text(position, value, f"{value:,}", ha="center", va="bottom")
    for result in results:
        axes[1, 0].plot(range(1, len(result.train_loss) + 1), result.train_loss, label=result.config.name)
        _plot_accuracy_curve(axes[1, 1], result, split=split, label=result.config.name)
    axes[1, 0].set_title("训练 Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Cross-Entropy")
    axes[1, 1].set_title(f"{evaluation_label} Accuracy（单次最终测试显示为点）")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[0, 1].text(0.98, 0.04, "训练时间：" + "｜".join(f"{name} {value:.2f}s" for name, value in zip(labels, times)), transform=axes[0, 1].transAxes, ha="right", fontsize=9)
    for axis in axes.ravel():
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"CNN 配置对比：共同使用{evaluation_label}集比较", fontsize=16)
    return fig


def format_result_table(results: list[TrainingResult]) -> str:
    split = _comparison_split(results)
    evaluation_label = "验证" if split == "val" else "测试"
    show_final_test = split == "val"
    heading = f"{'配置':<22}{evaluation_label + '准确率':>12}{evaluation_label + '右移':>12}"
    if show_final_test:
        heading += f"{'最终测试':>12}"
    heading += f"{'参数量':>12}{'耗时(秒)':>12}"
    lines = [f"CNN 配置对比（统一比较{evaluation_label}集）", "-" * 94, heading, "-" * 94]
    for result in results:
        line = f"{result.config.name:<22}{_comparison_accuracy(result, split):>11.1%}{shifted_accuracy(result, split=split):>11.1%}"
        if show_final_test:
            final_test = f"{result.final_test_accuracy:.1%}" if result.test_accuracy else "未评估"
            line += f"{final_test:>12}"
        line += f"{result.parameter_count:>12,}{result.elapsed_seconds:>12.2f}"
        lines.append(line)
    lines.append("-" * 94)
    return "\n".join(lines)


def print_result_table(results: list[TrainingResult]) -> None:
    print("\n" + format_result_table(results))
