"""不训练模型的卷积/池化小实验：把一个输出位置拆成可核算的数值。"""

from numbers import Integral, Real

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.nn import functional as F

from .plots import configure_chinese_font


def _matrix(values, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(values).detach().cpu()
        if tensor.is_complex():
            raise ValueError(f"{name}不能包含复数。")
        tensor = tensor.to(dtype=torch.float64).clone()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name}必须是由数值组成的二维矩阵。") from error
    if tensor.ndim != 2 or min(tensor.shape) < 1:
        raise ValueError(f"{name}必须是非空二维矩阵 [H,W]，当前 shape={tuple(tensor.shape)}。")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name}不能包含 NaN 或无穷大。")
    return tensor


def _integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        rule = "非负整数" if minimum == 0 else "正整数"
        raise ValueError(f"{name}必须是{rule}，当前值为 {value!r}。")
    return int(value)


def _output_size(height: int, width: int, kernel_size: int, padding: int, stride: int):
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    if min(output_height, output_width) < 1:
        raise ValueError(
            f"窗口 K={kernel_size} 放不进输入 {height}×{width}（padding={padding}）。"
            "请减小窗口、增加卷积 padding 或使用更大的输入矩阵。"
        )
    return output_height, output_width


def _position(value, output_shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("output_position 应写成 (行,列)，例如 (0,0)，位置从 0 开始。")
    row = _integer(value[0], "输出行", minimum=0)
    column = _integer(value[1], "输出列", minimum=0)
    if row >= output_shape[0] or column >= output_shape[1]:
        raise ValueError(
            f"输出位置 ({row},{column}) 超出了 {output_shape[0]}×{output_shape[1]} 输出；"
            f"行范围为 0–{output_shape[0]-1}，列范围为 0–{output_shape[1]-1}。"
            "改变 K/P/S 后请同步检查 output_position。"
        )
    return row, column


def _number(value: float) -> str:
    return f"{float(value):.5g}"


def _draw_matrix(axis, matrix: torch.Tensor, title: str, *, limits=None, mark=None):
    values = matrix.numpy()
    if limits is None:
        limits = (float(values.min()), float(values.max()))
    low, high = limits
    if low == high:
        low, high = low - 0.5, high + 0.5
    cmap = "coolwarm" if low < 0 else "Blues"
    picture = axis.imshow(values, cmap=cmap, vmin=low, vmax=high)
    height, width = values.shape
    # 小矩阵逐格标数；较大的图片保留坐标和色条，避免数值挤满画面。
    if max(height, width) <= 10:
        for row in range(height):
            for column in range(width):
                color = picture.cmap(picture.norm(values[row, column]))
                brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                axis.text(
                    column, row, _number(values[row, column]), ha="center", va="center",
                    fontsize=9 if max(height, width) <= 7 else 7,
                    color="white" if brightness < 0.5 else "black",
                )
        axis.set_xticks(np.arange(width))
        axis.set_yticks(np.arange(height))
        axis.set_xticks(np.arange(width + 1) - 0.5, minor=True)
        axis.set_yticks(np.arange(height + 1) - 0.5, minor=True)
        axis.grid(which="minor", color="white", linewidth=0.6)
        axis.tick_params(which="minor", length=0)
    else:
        axis.set_xticks(np.unique(np.linspace(0, width - 1, 5, dtype=int)))
        axis.set_yticks(np.unique(np.linspace(0, height - 1, 5, dtype=int)))
    if mark is not None:
        row, column, window_height, window_width = mark
        axis.add_patch(Rectangle(
            (column - 0.5, row - 0.5), window_width, window_height,
            fill=False, edgecolor="#E87900", linewidth=3,
        ))
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("列（从 0 开始）")
    axis.set_ylabel("行（从 0 开始）")
    axis.figure.colorbar(picture, ax=axis, shrink=0.65, label="数值")


def show_convolution_step(
    image, kernel=None, *, kernel_size=None, padding=0, stride=1,
    bias=0.0, output_position=(0, 0),
) -> dict:
    """展示单通道卷积的一个位置，并返回图、完整输出和手算中间量。

    image 是二维数值矩阵；kernel 为方形人工权重矩阵。未传 kernel 时使用
    K×K 全 1 核，未传 kernel_size 时从 kernel 推断 K（无 kernel 则 K=3）。
    本实验固定 dilation=1，只计算单图、单输入通道和单输出通道。
    padding 是四周补 0 的圈数，stride 是窗口移动步长，位置从 0 开始。
    PyTorch Conv2d 按给定核计算互相关，不翻转核；深度学习中通常称为卷积。
    """
    source = _matrix(image, "输入 image")
    padding = _integer(padding, "padding（补零圈数 P）", minimum=0)
    stride = _integer(stride, "stride（步长 S）")
    height, width = map(int, source.shape)
    if kernel is None:
        kernel_size = 3 if kernel_size is None else kernel_size
        kernel_size = _integer(kernel_size, "kernel_size（卷积核大小 K）")
        _output_size(height, width, kernel_size, padding, stride)
        weights = torch.ones(kernel_size, kernel_size, dtype=torch.float64)
    else:
        weights = _matrix(kernel, "卷积核 kernel")
        if weights.shape[0] != weights.shape[1]:
            raise ValueError("本页使用方形卷积核，请令 kernel 的行数与列数一致。")
        inferred = int(weights.shape[0])
        kernel_size = inferred if kernel_size is None else _integer(kernel_size, "kernel_size（卷积核大小 K）")
        if inferred != kernel_size:
            raise ValueError(f"kernel_size={kernel_size}，但 kernel 是 {inferred}×{inferred}。请同步修改卷积核。")
    if isinstance(bias, bool) or not isinstance(bias, Real) or not np.isfinite(bias):
        raise ValueError("bias 必须是有限的实数，例如 0 或 0.5。")
    output_shape = _output_size(height, width, kernel_size, padding, stride)
    row, column = _position(output_position, output_shape)
    padded = F.pad(source, (padding, padding, padding, padding))
    start_row, start_column = row * stride, column * stride
    window = padded[start_row:start_row + kernel_size, start_column:start_column + kernel_size]
    products = window * weights
    product_sum = float(products.sum())
    output = F.conv2d(
        source[None, None], weights[None, None],
        bias=torch.tensor([float(bias)], dtype=source.dtype), stride=stride, padding=padding,
    )[0, 0]
    chosen = float(output[row, column])
    height_formula = f"H_out = floor(({height} + 2×{padding} - {kernel_size}) / {stride}) + 1 = {output_shape[0]}"
    width_formula = f"W_out = floor(({width} + 2×{padding} - {kernel_size}) / {stride}) + 1 = {output_shape[1]}"
    print(f"输入 [B,C,H,W] = (1,1,{height},{width}) → 输出 (1,1,{output_shape[0]},{output_shape[1]})")
    print(height_formula)
    print(width_formula)
    print(f"Y[{row},{column}] = 逐元素乘积之和 {_number(product_sum)} + bias {_number(bias)} = {_number(chosen)}")

    configure_chinese_font()
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    input_limits = (float(padded.min()), float(padded.max()))
    _draw_matrix(axes[0, 0], padded, f"① 输入补零后｜{tuple(padded.shape)}\n橙框：计算 Y[{row},{column}] 的窗口", limits=input_limits,
                 mark=(start_row, start_column, kernel_size, kernel_size))
    _draw_matrix(axes[0, 1], window, f"② 取出 {kernel_size}×{kernel_size} 窗口\n补零后的起点 ({start_row},{start_column})", limits=input_limits)
    _draw_matrix(axes[0, 2], weights, f"③ 人工设置的卷积核 W｜{tuple(weights.shape)}")
    _draw_matrix(axes[1, 0], products, "④ 窗口 × 卷积核（逐元素）")
    axes[1, 1].axis("off")
    axes[1, 1].set_title("⑤ 求和、加偏置、核对尺寸", fontsize=11)
    axes[1, 1].text(
        0.0, 0.98,
        f"乘积之和 = {_number(product_sum)}\nbias = {_number(bias)}\n"
        f"Y[{row},{column}] = {_number(product_sum)} + {_number(bias)}\n"
        f"                 = {_number(chosen)}\n\n"
        f"K={kernel_size}，P={padding}，S={stride}\n"
        f"H_out = floor((H+2P−K)/S)+1\n"
        f"          = floor(({height}+2×{padding}−{kernel_size})/{stride})+1\n"
        f"          = {output_shape[0]}\n"
        f"W_out = {output_shape[1]}（宽度代入同一公式）\n\n"
        "橙框标出当前计算位置。\n输入与窗口共用色阶；其余看各自色条。\n尺寸公式固定 dilation=1。",
        va="top", fontsize=11, linespacing=1.35,
    )
    _draw_matrix(axes[1, 2], output, f"⑥ 完整输出 Y｜{tuple(output.shape)}\n橙框：Y[{row},{column}]={_number(chosen)}", mark=(row, column, 1, 1))
    fig.suptitle("一个卷积输出位置：取窗口 → 逐元素相乘 → 求和 + bias", fontsize=17)
    return {
        "figure": fig, "input": source, "padded_input": padded, "kernel": weights,
        "window": window, "products": products, "product_sum": product_sum,
        "bias": float(bias), "output": output, "output_value": chosen,
        "output_position": (row, column), "window_start": (start_row, start_column),
        "input_shape": (1, 1, height, width), "output_shape": (1, 1, *output_shape),
        "height_formula": height_formula, "width_formula": width_formula,
    }


def show_pooling_step(
    image, *, pooling="max", kernel_size=2, stride=None, output_position=(0, 0),
) -> dict:
    """显示单通道最大/平均池化的数值窗口；固定 padding=0、ceil_mode=False。

    stride=None 时步长等于窗口大小（PyTorch 的默认行为）。返回完整输出、
    所选窗口、max_value 和 mean_value，便于独立核对计算。
    """
    source = _matrix(image, "输入 image")
    if pooling not in ("max", "avg"):
        raise ValueError("pooling 请选择 'max'（最大池化）或 'avg'（平均池化）。")
    kernel_size = _integer(kernel_size, "kernel_size（池化窗口 K）")
    stride = kernel_size if stride is None else _integer(stride, "stride（池化步长 S）")
    height, width = map(int, source.shape)
    output_shape = _output_size(height, width, kernel_size, 0, stride)
    row, column = _position(output_position, output_shape)
    start_row, start_column = row * stride, column * stride
    window = source[start_row:start_row + kernel_size, start_column:start_column + kernel_size]
    max_value, mean_value = float(window.max()), float(window.mean())
    operation = F.max_pool2d if pooling == "max" else F.avg_pool2d
    output = operation(source[None, None], kernel_size=kernel_size, stride=stride)[0, 0]
    chosen = float(output[row, column])
    label = "最大池化 Max" if pooling == "max" else "平均池化 Average"
    height_formula = f"H_out = floor(({height} - {kernel_size}) / {stride}) + 1 = {output_shape[0]}"
    width_formula = f"W_out = floor(({width} - {kernel_size}) / {stride}) + 1 = {output_shape[1]}"
    print(f"{label}：输入 (1,1,{height},{width}) → 输出 (1,1,{output_shape[0]},{output_shape[1]})；padding=0")
    print(height_formula)
    print(width_formula)
    print(f"选中窗口：最大值={_number(max_value)}；均值={_number(float(window.sum()))}/{window.numel()}={_number(mean_value)}；当前输出={_number(chosen)}")

    configure_chinese_font()
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.8), constrained_layout=True)
    limits = (float(source.min()), float(source.max()))
    _draw_matrix(axes[0], source, f"① 输入特征图｜{tuple(source.shape)}\n橙框：窗口起点 ({start_row},{start_column})", limits=limits,
                 mark=(start_row, start_column, kernel_size, kernel_size))
    _draw_matrix(axes[1], window, f"② 取出 {kernel_size}×{kernel_size} 区域", limits=limits)
    axes[2].axis("off")
    axes[2].set_title("③ 两种局部汇总方法", fontsize=11)
    axes[2].text(
        0.0, 0.95,
        f"Max：最大值 = {_number(max_value)}\n\n"
        f"Average：总和 / 元素数\n"
        f"= {_number(float(window.sum()))} / {window.numel()}\n"
        f"= {_number(mean_value)}\n\n"
        f"本次选择：{label}\nY[{row},{column}] = {_number(chosen)}\n\n"
        f"K={kernel_size}，S={stride}，P=0\n"
        "H_out = floor((H−K)/S)+1\n"
        f"          = floor(({height}−{kernel_size})/{stride})+1\n"
        f"          = {output_shape[0]}\n"
        f"W_out = {output_shape[1]}\n\n"
        "各图共用色阶；池化不学习参数。",
        va="top", fontsize=10.5, linespacing=1.35,
    )
    _draw_matrix(axes[3], output, f"④ {label} 输出｜{tuple(output.shape)}\n橙框：当前输出位置", limits=limits, mark=(row, column, 1, 1))
    fig.suptitle("一个池化输出位置：同一局部区域，取最大值或求平均值", fontsize=16)
    return {
        "figure": fig, "input": source, "window": window, "output": output,
        "max_value": max_value, "mean_value": mean_value, "output_value": chosen,
        "output_position": (row, column), "window_start": (start_row, start_column),
        "input_shape": (1, 1, height, width), "output_shape": (1, 1, *output_shape),
        "height_formula": height_formula, "width_formula": width_formula,
    }
