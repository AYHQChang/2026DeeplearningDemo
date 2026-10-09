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
from .model import ComposableCNN, SmallCNN, _print_trace_rows, trace_forward


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
    common_max = max(float(feature.max()), 1e-6)
    for axis, tensor, title in zip(
        axes,
        (feature, max_pooled, avg_pooled),
        (f"卷积 + ReLU 后｜{data.image_size}×{data.image_size}", f"Max Pooling｜{data.image_size // 2}×{data.image_size // 2}", f"Average Pooling｜{data.image_size // 2}×{data.image_size // 2}"),
    ):
        axis.imshow(_image(tensor), cmap="magma", vmin=0, vmax=common_max)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("Pooling 汇总局部区域；三图共用色阶（暗=0，亮=较大正值）", fontsize=15)
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


def _plot_trace_features(rows: list[dict], max_channels: int, title: str) -> plt.Figure:
    """同一图内全部响应使用共同的对称色阶，保留负响应和零值。"""
    if type(max_channels) is not int or max_channels <= 0:
        raise ValueError("max_channels 必须是正整数。")
    configure_chinese_font()
    stages = [row for row in rows if row["tensor"].ndim == 4]
    shown = min(max_channels, max(row["tensor"].shape[1] for row in stages))
    responses = [row["tensor"][0, :shown] for row in stages if row["type"] != "Input"]
    limit = max([float(value.abs().max()) for value in responses] + [1e-6])
    fig, axes = plt.subplots(
        len(stages), shown + 1, figsize=(2.7 + 2.0 * shown, 2.05 * len(stages) + 0.6),
        constrained_layout=True, squeeze=False,
    )
    response_image = None
    for row_number, row in enumerate(stages):
        tensor = row["tensor"][0]
        count = min(shown, tensor.shape[0])
        axes[row_number, 0].axis("off")
        axes[row_number, 0].text(
            0.02, 0.94,
            f"{row_number:02d} {row['label']}\n{row['layer']}\n"
            f"输入 {row['input_shape']}\n输出 {row['output_shape']}\n"
            f"展示前 {count}/{tensor.shape[0]} 个通道\n"
            f"范围 [{float(tensor.min()):.3g}, {float(tensor.max()):.3g}]",
            ha="left", va="top", fontsize=9.5, linespacing=1.35,
        )
        for channel, axis in enumerate(axes[row_number, 1:]):
            axis.axis("off")
            if channel >= count:
                continue
            if row["type"] == "Input":
                axis.imshow(tensor[channel].numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                response_image = axis.imshow(
                    tensor[channel].numpy(), cmap="coolwarm", vmin=-limit, vmax=limit,
                )
            axis.set_title(f"通道 {channel}｜{tensor.shape[-2]}×{tensor.shape[-1]}", fontsize=10)
    if response_image is not None:
        colorbar = fig.colorbar(response_image, ax=axes[:, 1:].ravel().tolist(), fraction=0.025, pad=0.015)
        colorbar.set_label("共同响应色阶：蓝=负，浅灰=0，红=正", fontsize=10)
    fig.suptitle(
        title + "\n同一张图片，按真实执行顺序；输入用 0–1 灰度，所有响应共用色阶",
        fontsize=13,
    )
    return fig


def plot_forward_pass(
    model: torch.nn.Module, images: torch.Tensor, *,
    class_names: tuple[str, ...] | list[str] | None = None, max_channels: int = 4,
) -> tuple[plt.Figure, plt.Figure]:
    """观察第一张图片：逐层图 + 展平/分类图；不训练、不选择数据划分。

    从训练集或验证集传入图片即可。若 B>1，只观察 images[:1]，因此图中
    B=1。自定义网络的 nn.Module 被逐项捕获；函数式运算不会单独列行。
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or len(images) == 0:
        raise ValueError("images 应为非空 [B,C,H,W] Tensor。")
    rows = trace_forward(model, images[:1])
    logits = rows[-1]["tensor"]
    classes = logits.shape[1]
    if class_names is not None and len(class_names) != classes:
        raise ValueError(f"class_names 长度应等于模型输出类别数 {classes}。")
    names = [str(index) for index in range(classes)] if class_names is None else list(map(str, class_names))
    print("观察 images[:1]：本次 B=1，训练的 Batch size 可以不同。")
    _print_trace_rows(rows)
    if not isinstance(model, (SmallCNN, ComposableCNN)):
        print("自定义网络提示：仅逐项捕获 nn.Module；F.relu 等函数式运算不会单独列行，末行为实际模型输出。")
    probabilities = torch.softmax(logits, dim=1)
    prediction = int(probabilities.argmax(dim=1).item())
    print(f"Softmax：{tuple(logits.shape)} → {tuple(probabilities.shape)}；每张图片的概率之和 = {float(probabilities.sum()):.6f}")
    print(f"Argmax：{tuple(probabilities.shape)} → (1,)；预测标签 {prediction}，类别 {names[prediction]}")
    print("观察处于评估模式：Dropout 关闭、BatchNorm 使用累计统计；原训练状态已恢复。")
    features = _plot_trace_features(rows, max_channels, "逐层响应：卷积、激活和池化分别观察")
    head, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    flatten = next((row for row in reversed(rows) if row["type"] == "Flatten"), None)
    if flatten is not None:
        vector = flatten["tensor"][0]
        limit = max([
            float(row["tensor"][0, :max_channels].abs().max())
            for row in rows if row["tensor"].ndim == 4 and row["type"] != "Input"
        ] + [1e-6])
        axes[0].imshow(vector.numpy()[None, :], cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
        axes[0].set_box_aspect(0.25)
        axes[0].set_yticks([])
        axes[0].set_xlabel(
            "特征索引；一维向量压成一行，沿用响应色阶\n前 8 个值："
            + ", ".join(f"{float(value):.3g}" for value in vector[:8]), fontsize=9,
        )
        axes[0].set_title(f"Flatten：{flatten['input_shape']}\n→ {flatten['output_shape']}；F={vector.numel()}", fontsize=11)
    else:
        axes[0].axis("off")
        axes[0].text(0.05, 0.75, "未捕获到 nn.Flatten 模块。\n自定义网络可能使用函数式展平\n或其他分类头。", va="top", fontsize=12)
    # 类别较多时展示概率最高的 12 类；预测仍根据全部类别计算。
    indices = probabilities[0].topk(min(classes, 12)).indices.sort().values
    positions = np.arange(len(indices))
    shown_names = [names[int(index)] for index in indices]
    axes[1].bar(positions, logits[0, indices].numpy(), color="#7C3AED")
    axes[1].axhline(0, color="#475569", linewidth=0.8)
    axes[1].set_title(f"全连接输出 logits：{tuple(logits.shape)}\n原始分数，可以为负，也不要求总和为 1", fontsize=11)
    axes[2].bar(positions, probabilities[0, indices].numpy(), color="#2563EB")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title(f"Softmax 概率：{tuple(probabilities.shape)}\n全部 {classes} 类概率总和为 1；argmax → [B]", fontsize=11)
    for axis in axes[1:]:
        axis.set_xticks(positions, shown_names, rotation=40, ha="right", fontsize=9)
        axis.set_xlabel("类别" + ("（只展示概率最高的 12 类）" if classes > 12 else ""))
        axis.grid(axis="y", alpha=0.2)
    head.suptitle(f"从四维响应到分类分数向量：预测 {names[prediction]}（标签 {prediction}）", fontsize=14)
    return features, head


def plot_feature_maps(
    result: TrainingResult, sample_index: int = 0, max_channels: int = 8,
    split: str = "auto",
) -> plt.Figure:
    """内置模型逐项展示真实阶段；保留自定义 feature_maps() 兼容入口。"""
    images, truth, evaluation_label = evaluation_data(result, split=split)
    if not 0 <= sample_index < len(images):
        raise IndexError(f"sample_index 必须在 0 到 {len(images) - 1} 之间。")
    sample = images[sample_index:sample_index + 1]
    rows = trace_forward(result.model, sample)
    prediction = int(rows[-1]["tensor"].argmax(dim=1).item())
    if not isinstance(result.model, (SmallCNN, ComposableCNN)):
        feature_maps = getattr(result.model, "feature_maps", None)
        if callable(feature_maps):
            modes = {module: module.training for module in result.model.modules()}
            parameter = next(result.model.parameters(), None)
            device = parameter.device if parameter is not None else sample.device
            dtype = parameter.dtype if parameter is not None else sample.dtype
            try:
                result.model.eval()
                with torch.no_grad():
                    maps = feature_maps(sample.to(device=device, dtype=dtype))
            finally:
                for module, training in modes.items():
                    module.training = training
            if not isinstance(maps, dict) or not maps:
                raise ValueError("feature_maps() 应返回非空字典：阶段名 -> [B,C,H,W] Tensor。")
            feature_rows = [rows[0]]
            for name, tensor in maps.items():
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or len(tensor) != 1:
                    raise ValueError(f"特征图 {name} 应为 [1,C,H,W] Tensor。")
                feature_rows.append({
                    "layer": str(name), "label": str(name), "type": "FeatureMap",
                    "input_shape": (), "output_shape": tuple(tensor.shape),
                    "tensor": tensor.detach().cpu().clone(),
                })
            rows = feature_rows
    return _plot_trace_features(
        rows, max_channels,
        f"{evaluation_label}样本｜真实 {_class_name(result.data, int(truth[sample_index]))} / 预测 {_class_name(result.data, prediction)}",
    )


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
