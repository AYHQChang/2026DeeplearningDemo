"""不打开 Notebook 时使用的最短 Transformer 演示入口。"""

import argparse

from transformer_lab import compare, quick_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="小型 Transformer 注意力寻宝实验室")
    parser.add_argument("--experiment", choices=("baseline", "comparison"), default="baseline")
    parser.add_argument("--device", default="cpu", help='cpu 或课堂分配的 "cuda:编号"')
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.experiment == "comparison":
        compare(epochs=min(args.epochs, 12), device=args.device)
    else:
        quick_demo(epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
