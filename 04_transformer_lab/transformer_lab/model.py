"""适合课堂阅读的小型 Transformer Encoder 分类模型。"""

import math

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    choices = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}
    if name not in choices:
        raise ValueError(f"activation 只能是 {tuple(choices)}。")
    return choices[name]


class SinusoidalPositionalEncoding(nn.Module):
    """无可训练参数的正弦位置编码。"""

    def __init__(self, d_model: int, max_length: int = 25) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError("输入序列超过位置编码的 max_length。")
        return x + self.encoding[:, : x.shape[1]]


class TinyEncoderBlock(nn.Module):
    """一层 Self-Attention、残差连接、LayerNorm 和 Feed-forward。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attention(
            x, x, x, need_weights=need_weights, average_attn_weights=False
        )
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.feedforward(x)))
        return x, weights


class TinyTransformerClassifier(nn.Module):
    """读取末尾 QUERY token 的表示，输出三分类 logits。"""

    def __init__(
        self,
        vocab_size: int = 4,
        n_classes: int = 3,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_positional_encoding: bool = True,
        max_length: int = 25,
    ) -> None:
        super().__init__()
        if min(vocab_size, n_classes, d_model, num_heads, num_layers, dim_feedforward, max_length) <= 0:
            raise ValueError("词表、类别数、模型维度、Head、层数、Feed-forward 和长度必须大于 0。")
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_length=max_length)
        self.layers = nn.ModuleList(
            TinyEncoderBlock(d_model, num_heads, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def encode(
        self, token_ids: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        if self.use_positional_encoding:
            x = self.position(x)
        last_weights = None
        for layer in self.layers:
            x, last_weights = layer(x, need_weights=need_weights)
        return x, last_weights

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encode(token_ids)
        return self.classifier(encoded[:, -1])

    def forward_with_attention(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, weights = self.encode(token_ids, need_weights=True)
        if weights is None:
            raise RuntimeError("模型没有可返回的 Attention 权重。")
        return self.classifier(encoded[:, -1]), weights


def scaled_dot_product_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """用于手算核对的单个 Query 注意力，不代替 PyTorch 多头实现。"""

    if query.ndim != 1 or keys.ndim != 2 or values.ndim != 2:
        raise ValueError("query 应为 [D]，keys 和 values 应为 [T,D]。")
    if keys.shape[0] != values.shape[0] or query.shape[0] != keys.shape[1]:
        raise ValueError("Q、K、V 的 shape 不匹配。")
    scores = keys @ query / math.sqrt(query.numel())
    weights = scores.softmax(dim=0)
    return scores, weights, weights @ values


def estimate_parameter_count(
    d_model: int,
    num_layers: int,
    dim_feedforward: int,
    vocab_size: int = 4,
    n_classes: int = 3,
) -> int:
    """在分配模型前计算本项目固定结构的精确参数量。"""

    if min(d_model, num_layers, dim_feedforward, vocab_size, n_classes) <= 0:
        raise ValueError("参数量估计所需的维度必须大于 0。")
    one_layer = 4 * d_model**2 + 2 * d_model * dim_feedforward + dim_feedforward + 9 * d_model
    return vocab_size * d_model + num_layers * one_layer + d_model * n_classes + n_classes
