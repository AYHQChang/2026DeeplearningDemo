"""不打开 Notebook 时使用的可调 RNN 演示入口。"""

import argparse

from rnn_lab import quick_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="RNN 延迟回忆课堂实验室")
    parser.add_argument("--device", default="cpu", help="cpu 或课堂分配的 cuda:编号")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--cell-type", choices=("rnn", "lstm", "gru"), default="lstm")
    parser.add_argument("--sequence-length", type=int, default=30)
    args = parser.parse_args()
    quick_demo(
        cell_type=args.cell_type,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
