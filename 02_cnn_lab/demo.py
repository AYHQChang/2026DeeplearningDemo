"""CNN 实验室的简洁命令行入口。"""

import argparse

from cnn_lab import compare_pooling, quick_demo, resolve_device, show_convolution, show_data


def _channels(text: str) -> tuple[int, int]:
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("channels 应写成 8,16。") from error
    if len(values) != 2 or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("channels 必须包含两个正整数。")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CNN 手写数字课堂实验室")
    parser.add_argument("--experiment", choices=("gallery", "mechanism", "baseline", "pooling"), default="baseline")
    parser.add_argument("--device", default="cpu", help="cpu、auto、cuda 或课堂分配的 cuda:编号")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--channels", type=_channels, default=(8, 16))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"运行设备：{resolve_device(args.device)}")
    if args.experiment == "gallery":
        show_data()
    elif args.experiment == "mechanism":
        show_convolution()
    elif args.experiment == "pooling":
        compare_pooling(channels=args.channels, epochs=args.epochs, device=args.device)
    else:
        quick_demo(channels=args.channels, epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
