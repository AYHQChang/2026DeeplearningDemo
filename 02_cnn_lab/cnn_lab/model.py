"""适合课堂阅读、并可直接扩展到 28×28 输入的小型 CNN。"""

import torch
from torch import nn
from torch.nn import functional as F


class SmallCNN(nn.Module):
    """两层卷积网络；空间尺寸由 `image_size` 计算，不写死为 8×8。"""

    def __init__(
        self,
        image_size: int,
        n_classes: int = 10,
        channels: tuple[int, int] = (8, 16),
        kernel_size: int = 3,
        pooling: str = "max",
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

        first = F.relu(self.conv1(x))
        pooled = self.pool(first)
        second = F.relu(self.conv2(pooled))
        return {"第一层卷积": first, "池化后": pooled, "第二层卷积": second}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = self.feature_maps(x)
        return self.classifier(maps["第二层卷积"])
