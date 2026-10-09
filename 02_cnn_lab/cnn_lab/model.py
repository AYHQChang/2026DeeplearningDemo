"""适合课堂阅读、并可直接扩展到 28×28 输入的小型 CNN。"""

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _activation(name: str):
    """把教学配置中的名称转换为 PyTorch 激活函数。"""
    choices = {"relu": F.relu, "tanh": torch.tanh, "leaky_relu": F.leaky_relu}
    if name not in choices:
        raise ValueError(f"activation 只能是：{'、'.join(choices)}。")
    return choices[name]


class SmallCNN(nn.Module):
    """两层卷积网络；空间尺寸由 `image_size` 计算，不写死为 8×8。"""

    def __init__(
        self,
        image_size: int,
        n_classes: int = 10,
        channels: tuple[int, int] = (8, 16),
        kernel_size: int = 3,
        pooling: str = "max",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size < 4:
            raise ValueError("image_size 至少为 4。")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须是正奇数。")
        if len(channels) != 2 or any(value <= 0 for value in channels):
            raise ValueError("channels 必须包含两个正整数。")
        if pooling not in {"max", "avg", "none"}:
            raise ValueError("pooling 只能是 max、avg 或 none。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")

        padding = kernel_size // 2
        # 注册为 Module，使同一次真实 forward 中的两次激活都能被观察。
        self.activation = _ConfiguredActivation(activation)
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size, padding=padding)
        self.pool = {
            "max": nn.MaxPool2d(2),
            "avg": nn.AvgPool2d(2),
            "none": nn.Identity(),
        }[pooling]
        pooled_size = image_size // 2 if pooling != "none" else image_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[1] * pooled_size * pooled_size, n_classes),
        )

    def feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """保留旧接口：两个“卷积”键实际为激活后的结果。

        逐项区分卷积、激活与池化时，请使用 trace_forward()。
        """

        first = self.activation(self.conv1(x))
        pooled = self.pool(first)
        second = self.activation(self.conv2(pooled))
        return {"第一层卷积": first, "池化后": pooled, "第二层卷积": second}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = self.feature_maps(x)
        return self.classifier(maps["第二层卷积"])


@dataclass(frozen=True)
class ConvBlock:
    """一块的顺序：卷积 → 可选 BatchNorm → 激活 → 可选池化 → 可选 Dropout。"""

    out_channels: int = 8
    kernel_size: int = 3
    activation: str = "relu"
    pooling: str = "none"
    batch_norm: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if type(self.out_channels) is not int or not 1 <= self.out_channels <= 64:
            raise ValueError("每个卷积块的 out_channels 必须是 1–64 的整数。")
        if type(self.kernel_size) is not int or self.kernel_size not in {1, 3, 5, 7}:
            raise ValueError("每个卷积块的 kernel_size 只能是 1、3、5、7。")
        _activation(self.activation)
        if self.pooling not in {"max", "avg", "none"}:
            raise ValueError("每个卷积块的 pooling 只能是 max、avg 或 none。")
        if type(self.batch_norm) is not bool:
            raise ValueError("batch_norm 必须填写 True 或 False。")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("卷积块 dropout 必须位于 [0, 1) 区间。")


class _ConfiguredActivation(nn.Module):
    """沿用同一个激活映射；修改 _activation() 后两种 CNN 都能使用。"""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.function = _activation(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.function(x)

    def extra_repr(self) -> str:
        return self.name


class ComposableCNN(nn.Module):
    """用 1–3 个可编辑卷积块搭建网络，分类层随输入尺寸和类别数计算。"""

    def __init__(
        self,
        image_size: int,
        n_classes: int,
        blocks: tuple[ConvBlock, ...],
        input_channels: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if type(image_size) is not int or image_size < 4:
            raise ValueError("image_size 必须是至少为 4 的整数。")
        if type(n_classes) is not int or n_classes < 2:
            raise ValueError("n_classes 必须是至少为 2 的整数。")
        if type(input_channels) is not int or input_channels < 1:
            raise ValueError("input_channels 必须是正整数。")
        if not 1 <= len(blocks) <= 3:
            raise ValueError("blocks 必须包含 1–3 个 ConvBlock。")
        if not all(isinstance(block, ConvBlock) for block in blocks):
            raise ValueError("blocks 中的每一项都应写为 ConvBlock(...)。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("分类头 dropout 必须位于 [0, 1) 区间。")

        self.specs = tuple(blocks)
        self.blocks = nn.ModuleList()
        channels = input_channels
        size = image_size
        for number, spec in enumerate(self.specs, start=1):
            layers = OrderedDict()
            layers["conv"] = nn.Conv2d(
                channels, spec.out_channels, spec.kernel_size,
                padding=spec.kernel_size // 2,
            )
            if spec.batch_norm:
                layers["batch_norm"] = nn.BatchNorm2d(spec.out_channels)
            layers["activation"] = _ConfiguredActivation(spec.activation)
            if spec.pooling != "none":
                if size < 2:
                    raise ValueError(
                        f"第 {number} 个卷积块池化前尺寸只有 {size}×{size}；"
                        "请减少池化次数，或把该块 pooling 改为 none。"
                    )
                layers["pooling"] = (
                    nn.MaxPool2d(2) if spec.pooling == "max" else nn.AvgPool2d(2)
                )
                size //= 2
            if spec.dropout:
                layers["dropout"] = nn.Dropout2d(spec.dropout)
            self.blocks.append(nn.Sequential(layers))
            channels = spec.out_channels
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(channels * size * size, n_classes),
        )

    def feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """显示每个完整卷积块的输出（包括该块启用的池化等组件）。"""
        maps = {}
        for number, block in enumerate(self.blocks, start=1):
            x = block(x)
            maps[f"卷积块 {number}"] = x
        return maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)


def _stage_description(module: nn.Module) -> tuple[str, str]:
    """名称来自真实 Module；不根据预设图片尺寸推测输出。"""
    if isinstance(module, nn.Conv2d):
        return "卷积（原始输出）", (
            f"C: {module.in_channels}→{module.out_channels}；"
            f"kernel={module.kernel_size}, stride={module.stride}, padding={module.padding}；"
            f"权重形状 {tuple(module.weight.shape)}"
        )
    if isinstance(module, _ConfiguredActivation):
        return f"激活 {module.name}", "逐元素计算，shape 不变"
    if isinstance(module, (nn.ReLU, nn.Tanh, nn.LeakyReLU, nn.Sigmoid)):
        return f"激活 {type(module).__name__}", "逐元素计算，shape 不变"
    if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
        name = "最大池化" if isinstance(module, nn.MaxPool2d) else "平均池化"
        return name, (
            f"kernel={module.kernel_size}, stride={module.stride}, padding={module.padding}；"
            "汇总空间窗口，通道数不变"
        )
    if isinstance(module, nn.BatchNorm2d):
        return "BatchNorm（评估模式）", "使用累计均值/方差；不更新统计量，shape 不变"
    if isinstance(module, (nn.Dropout, nn.Dropout2d)):
        return "Dropout（评估时关闭）", f"p={module.p:g}；观察时保留全部值，训练时才随机屏蔽"
    if isinstance(module, nn.Flatten):
        return "Flatten 展平", f"合并维度 {module.start_dim} 到 {module.end_dim}；不学习参数"
    if isinstance(module, nn.Linear):
        return "全连接", f"每张图片的 {module.in_features} 个特征 → {module.out_features} 个输出分数"
    if isinstance(module, nn.Identity):
        return "保持原值（不池化）", "shape 和数值均不变"
    return type(module).__name__, "按该模块的 forward 计算"


def trace_forward(model: nn.Module, images: torch.Tensor) -> list[dict]:
    """运行一次真实 forward，记录输入与每次叶子模块调用。

    观察统一用 eval/no_grad，并恢复每个模块原有的训练状态。保存的
    Tensor 是独立 CPU 副本，不保留训练图。自定义网络的函数式运算
    （如 F.relu）不会单独触发模块 hook，但末行始终对应实际模型输出。
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images 应是非空的 [B,C,H,W] Tensor；单张图片也要保留 B 轴。")
    if any(size <= 0 for size in images.shape) or not images.is_floating_point():
        raise ValueError("images 的 B、C、H、W 应为正数，像素使用浮点 Tensor。")
    parameter = next(model.parameters(), None)
    buffer = next(model.buffers(), None)
    reference = parameter if parameter is not None else buffer
    device = reference.device if reference is not None else images.device
    dtype = reference.dtype if reference is not None and reference.is_floating_point() else images.dtype
    sample = images.detach().to(device=device, dtype=dtype)
    rows = [{
        "layer": "input", "label": "输入图片", "type": "Input",
        "input_shape": tuple(sample.shape), "output_shape": tuple(sample.shape),
        "parameters": 0, "details": "[B,C,H,W]：批量、通道、高、宽",
        "tensor": sample.cpu().clone(),
    }]
    handles = []
    modes = {module: module.training for module in model.modules()}
    counted_parameters = set()
    calls = {}

    def hook(name, module):
        def record(_module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            tensor_input = next((value for value in inputs if isinstance(value, torch.Tensor)), None)
            calls[name] = calls.get(name, 0) + 1
            layer = name if calls[name] == 1 else f"{name}#{calls[name]}"
            label, details = _stage_description(module)
            parameters = 0
            for value in module.parameters(recurse=False):
                if id(value) not in counted_parameters:
                    parameters += value.numel()
                    counted_parameters.add(id(value))
            rows.append({
                "layer": layer, "label": label, "type": type(module).__name__,
                "input_shape": tuple(tensor_input.shape) if tensor_input is not None else (),
                "output_shape": tuple(output.shape), "parameters": parameters,
                "details": details, "tensor": output.detach().cpu().clone(),
            })
        return record

    try:
        for name, module in model.named_modules():
            if not list(module.children()):
                handles.append(module.register_forward_hook(hook(name or "model", module)))
        model.eval()
        with torch.no_grad():
            output = model(sample)
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != len(sample):
            raise ValueError("分类模型应返回 [B,类别数] 的原始分数 Tensor。")
        actual = output.detach().cpu().clone()
        if rows[-1]["output_shape"] != tuple(actual.shape) or not torch.equal(rows[-1]["tensor"], actual):
            rows.append({
                "layer": "model_output", "label": "模型实际输出（含函数式运算）",
                "type": "ModelOutput", "input_shape": rows[-1]["output_shape"],
                "output_shape": tuple(actual.shape), "parameters": 0,
                "details": "模块 hook 之外的运算合并到实际输出；详细步骤请写成 nn.Module",
                "tensor": actual,
            })
        rows[-1]["details"] += "；分类 logits，尚未转成概率"
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes.items():
            module.training = training
    return rows


def _print_trace_rows(rows: list[dict]) -> None:
    print("执行顺序｜层（源码名称）｜输入 shape → 输出 shape｜参数量")
    for number, row in enumerate(rows):
        print(
            f"{number:02d} {row['label']} ({row['layer']})\n"
            f"   {row['input_shape']} → {row['output_shape']}；参数 {row['parameters']:,}\n"
            f"   {row['details']}"
        )


def print_model_summary(
    model: nn.Module, image_size: int, input_channels: int = 1
) -> list[dict]:
    """复用真实执行记录；预览的 B=1 不限制训练时的 Batch size。"""
    rows = trace_forward(model, torch.zeros(1, input_channels, image_size, image_size))
    print("结构预览使用一张零值图片（B=1）；这是 shape 检查，数值不代表真实图片响应。")
    _print_trace_rows(rows)
    print(f"总参数量：{sum(p.numel() for p in model.parameters()):,}")
    # 保留轻量的旧返回接口；真实图片数值由 trace_forward() 返回。
    return [{key: value for key, value in row.items() if key != "tensor"} for row in rows]
