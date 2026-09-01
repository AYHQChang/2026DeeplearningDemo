"""MLP 模型结构：完成参数练习后，从这个短文件开始阅读和修改。"""

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    """把名称转换成对应的 PyTorch 激活层。"""
    choices = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in choices:
        raise ValueError("activation 只能是 relu、tanh 或 sigmoid。")
    return choices[name]


class MLP(nn.Module):
    """二维输入、多层隐藏层和分类 logits 输出组成的 MLP。"""

    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        activation: str,
        dropout: float,
        n_classes: int,
    ) -> None:
        super().__init__()
        activation_class = _activation(activation)
        sizes = (2, *hidden_sizes, n_classes)
        layers: list[nn.Module] = []

        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            is_output_layer = index == len(sizes) - 2
            if not is_output_layer:
                layers.append(activation_class())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        # 输出层直接给出 logits；CrossEntropyLoss 前不需要添加 Softmax。
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
