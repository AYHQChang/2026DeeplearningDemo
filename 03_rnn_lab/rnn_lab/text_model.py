"""教师参考模型集合：完整单向 RNN/LSTM/GRU 与无顺序基线。

保留旧示例、图和复现接口；学生三级实验走 text_course，后两级须显式传入自己模型。
"""

import torch
from torch import nn


def valid_last(outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """取最后有效位置；右侧 PAD 末尾不属于评论。"""
    _check_lengths(outputs, lengths)
    return outputs[torch.arange(len(outputs), device=outputs.device), lengths.to(outputs.device) - 1]


def masked_mean(outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """只沿有效时间轴平均，不平均 batch 或特征。"""
    _check_lengths(outputs, lengths)
    mask = torch.arange(outputs.shape[1], device=outputs.device)[None, :] < lengths.to(outputs.device)[:, None]
    return (outputs * mask.unsqueeze(-1)).sum(dim=1) / lengths.to(outputs.device).unsqueeze(1)


def _check_lengths(outputs, lengths):
    if outputs.ndim != 3 or min(outputs.shape) < 1 or lengths.ndim != 1 or len(lengths) != len(outputs):
        raise ValueError("输出应为非空 [B,T,H]，lengths 应为 [B]。")
    if lengths.dtype != torch.int64 or int(lengths.min()) < 1 or int(lengths.max()) > outputs.shape[1]:
        raise ValueError("lengths 应为 int64 且位于 1 到 T。")


class TextClassifier(nn.Module):
    """不使用双向网络；padding_idx 只作用于字向量，不会跳过循环 PAD。"""

    def __init__(self, vocab_size: int, n_classes: int, embedding_dim: int = 32,
                 hidden_size: int = 32, cell_type: str = "gru", num_layers: int = 1,
                 dropout: float = 0.0, readout: str = "last"):
        super().__init__()
        if min(vocab_size, n_classes, embedding_dim, hidden_size, num_layers) < 1 or n_classes < 2:
            raise ValueError("词表、维度和层数必须为正，类别数至少为 2。")
        if cell_type not in {"rnn", "lstm", "gru", "mean_embedding"} or readout not in {"last", "mean"}:
            raise ValueError("cell_type 为 rnn/lstm/gru/mean_embedding；readout 为 last/mean。")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须位于 [0,1)。")
        self.cell_type, self.readout = cell_type, readout
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.recurrent = None
        if cell_type != "mean_embedding":
            recurrent_class = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell_type]
            # PyTorch 的循环 dropout 仅位于堆叠层之间；单层实际为 0。
            self.recurrent = recurrent_class(embedding_dim, hidden_size, num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(embedding_dim if self.recurrent is None else hidden_size, n_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.dtype != torch.int64 or min(token_ids.shape) < 1:
            raise ValueError("token_ids 必须是非空 int64 [B,T]。")
        embedded = self.embedding(token_ids)
        _check_lengths(embedded, lengths)
        if self.recurrent is None:
            sentence = masked_mean(embedded, lengths)
        else:
            outputs, _ = self.recurrent(embedded)
            sentence = valid_last(outputs, lengths) if self.readout == "last" else masked_mean(outputs, lengths)
        return self.classifier(sentence)
