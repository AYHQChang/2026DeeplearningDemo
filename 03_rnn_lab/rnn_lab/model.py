"""适合课堂阅读的短小 RNN、LSTM、GRU 序列分类模型。"""

import torch
from torch import nn


class RecurrentClassifier(nn.Module):
    """用序列最后一个输出完成分类，循环单元可在三种结构间切换。"""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 24,
        n_classes: int = 3,
        cell_type: str = "rnn",
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        cell_type = cell_type.lower()
        if cell_type not in {"rnn", "lstm", "gru"}:
            raise ValueError("cell_type 只能是 rnn、lstm 或 gru。")
        if min(input_size, hidden_size, n_classes, num_layers) <= 0:
            raise ValueError("input_size、hidden_size、n_classes 和 num_layers 必须大于 0。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")

        recurrent_class = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell_type]
        # PyTorch 只在循环层之间使用 dropout；单层网络没有层间连接。
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.cell_type = cell_type
        self.recurrent = recurrent_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        if cell_type == "lstm":
            # 让遗忘门初始时更愿意保留信息，减少课堂小样本训练的随机失败。
            for name, parameter in self.recurrent.named_parameters():
                if "bias_hh" in name:
                    start, end = hidden_size, 2 * hidden_size
                    with torch.no_grad():
                        parameter[start:end].fill_(1.0)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def sequence_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回每个时间步的隐藏表示，形状为 ``[B, T, H]``。"""

        outputs, _ = self.recurrent(x)
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """读取最后时间步的隐藏表示，输出 ``[B, classes]`` logits。"""

        return self.classifier(self.sequence_features(x)[:, -1, :])
