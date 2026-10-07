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
        self.activation = _activation(activation)
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
        """返回两层激活和池化结果，供 Notebook 观察。"""

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


def print_model_summary(
    model: nn.Module, image_size: int, input_channels: int = 1
) -> list[dict]:
    """用一张零值图片检查各层输出，不更新 BatchNorm 或触发 Dropout。"""
    rows = []
    handles = []
    modes = {module: module.training for module in model.modules()}
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32

    def hook(name, module):
        def record(_module, _inputs, output):
            rows.append({
                "layer": name, "type": type(module).__name__,
                "output_shape": tuple(output.shape),
                "parameters": sum(p.numel() for p in module.parameters(recurse=False)),
            })
        return record

    try:
        for name, module in model.named_modules():
            if not list(module.children()):
                handles.append(module.register_forward_hook(hook(name or "model", module)))
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, input_channels, image_size, image_size, device=device, dtype=dtype))
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes.items():
            module.training = training
    print(f"{'层':<32} {'输出尺寸 [B, ...]':<24} 参数量")
    for row in rows:
        print(f"{row['layer']:<32} {str(row['output_shape']):<24} {row['parameters']:,}")
    print(f"总参数量：{sum(p.numel() for p in model.parameters()):,}")
    return rows
