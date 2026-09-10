"""用同一配置比较四个课堂 Demo 的 CPU / 指定 GPU 训练速度。

示例：

    python device_speed_demo.py --lab cnn --devices cpu,cuda:0
    python device_speed_demo.py --lab transformer --devices cpu,cuda:0 --preview-epochs 2

``--preview-epochs`` 只实跑少量 Epoch，并按每 Epoch 平均训练时间粗略估算
完整目标；预览 Accuracy 不能当作模型最终效果。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
from pathlib import Path
import re
import sys
import time
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parent
LAB_DIRS = {
    "mlp": "01_mlp_lab",
    "cnn": "02_cnn_lab",
    "rnn": "03_rnn_lab",
    "transformer": "04_transformer_lab",
}
DEFAULT_EPOCHS = {"mlp": 40, "cnn": 20, "rnn": 12, "transformer": 12}
MAX_TARGET_EPOCHS = {"mlp": 100, "cnn": 40, "rnn": 40, "transformer": 30}


def parse_devices(text: str) -> tuple[str, ...]:
    """解析 ``cpu``、``cuda:x`` 或逗号分隔的两者，拒绝自动选卡。"""

    devices = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    if not devices or len(devices) > 2 or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("devices 应为 cpu、cuda:x 或 cpu,cuda:x。")
    if any(device != "cpu" and not re.fullmatch(r"cuda:\d+", device) for device in devices):
        raise argparse.ArgumentTypeError("只允许 cpu 或教师分配的 cuda:x；不允许 auto 和裸 cuda。")
    if len(devices) == 2 and "cpu" not in devices:
        raise argparse.ArgumentTypeError("双设备对比只允许 cpu,cuda:x，不负责多 GPU 调度。")
    return devices


def _lab_runner(lab: str) -> Callable[[str, int, int], tuple[float, float, int]]:
    """复用所选实验室的训练 API，返回训练秒数、Accuracy 和参数量。"""

    lab_path = str(ROOT / LAB_DIRS[lab])
    if lab_path not in sys.path:
        sys.path.insert(0, lab_path)

    if lab == "mlp":
        from mlp_lab import base_config, make_dataset, train_experiment

        def run(device: str, epochs: int, seed: int) -> tuple[float, float, int]:
            config = replace(
                base_config(dataset="moons", mode="fast", seed=seed, device=device),
                name="CPU/GPU 速度演示", hidden_sizes=(16, 16),
                epochs=epochs, n_samples=360,
            )
            data = make_dataset(config.dataset, config.n_samples, config.noise, config.seed)
            result = train_experiment(config, data=data)
            return result.elapsed_seconds, result.final_test_accuracy, result.parameter_count

        return run

    if lab == "cnn":
        from cnn_lab import base_config, train_experiment

        def run(device: str, epochs: int, seed: int) -> tuple[float, float, int]:
            config = replace(base_config(seed=seed, device=device), name="CPU/GPU 速度演示", epochs=epochs)
            result = train_experiment(config)
            return result.elapsed_seconds, result.final_test_accuracy, result.parameter_count

        return run

    if lab == "rnn":
        from rnn_lab import ExperimentConfig, run_experiment

        def run(device: str, epochs: int, seed: int) -> tuple[float, float, int]:
            result = run_experiment(ExperimentConfig(
                name="CPU/GPU 速度演示", cell_type="lstm", sequence_length=30,
                hidden_size=24, epochs=epochs, seed=seed, device=device,
            ))
            return result["elapsed_seconds"], result["metrics"]["accuracy"], result["parameter_count"]

        return run

    from transformer_lab import ExperimentConfig, run_experiment

    def run(device: str, epochs: int, seed: int) -> tuple[float, float, int]:
        result = run_experiment(ExperimentConfig(
            name="CPU/GPU 速度演示", epochs=epochs, seed=seed, device=device,
        ))
        return result["elapsed_seconds"], result["metrics"]["accuracy"], result["parameter_count"]

    return run


def _cuda_index(device: str) -> int | None:
    if device == "cpu":
        return None
    index = int(device.split(":", 1)[1])
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError(f"{device} 不可用；当前可见 GPU 数量为 {torch.cuda.device_count()}。")
    return index


def _run_once(
    runner: Callable[[str, int, int], tuple[float, float, int]],
    device: str,
    actual_epochs: int,
    target_epochs: int,
    seed: int,
) -> dict[str, float | int | str | None]:
    index = _cuda_index(device)
    if index is not None:
        torch.cuda.set_device(index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(index)
    else:
        index = None

    wall_start = time.perf_counter()
    train_seconds, accuracy, parameters = runner(device, actual_epochs, seed)
    if index is not None:
        torch.cuda.synchronize(index)
    wall_seconds = time.perf_counter() - wall_start
    per_epoch = train_seconds / actual_epochs
    summary: dict[str, float | int | str | None] = {
        "device": device,
        "device_name": torch.cuda.get_device_name(index) if index is not None else "CPU（单线程）",
        "actual_epochs": actual_epochs,
        "target_epochs": target_epochs,
        "train_seconds": train_seconds,
        "wall_seconds": wall_seconds,
        "seconds_per_epoch": per_epoch,
        "estimated_total_seconds": per_epoch * target_epochs,
        "estimated_remaining_seconds": per_epoch * (target_epochs - actual_epochs),
        "accuracy": accuracy,
        "parameters": parameters,
        "peak_memory_mb": torch.cuda.max_memory_allocated(index) / 1024**2 if index is not None else None,
    }
    gc.collect()
    if index is not None:
        torch.cuda.empty_cache()
    return summary


def _print_summary(summary: dict[str, float | int | str | None]) -> None:
    print(f"\n[{summary['device']}] {summary['device_name']}")
    print(f"实跑 Epoch：{summary['actual_epochs']}/{summary['target_epochs']}｜参数量：{summary['parameters']:,}")
    print(f"训练时间：{summary['train_seconds']:.3f} 秒｜每 Epoch：{summary['seconds_per_epoch']:.3f} 秒")
    print(f"端到端时间：{summary['wall_seconds']:.3f} 秒（含数据、模型构造和最终评估，不含绘图）")
    if summary["peak_memory_mb"] is not None:
        print(f"GPU 峰值已分配显存：{summary['peak_memory_mb']:.1f} MiB")
    if summary["actual_epochs"] < summary["target_epochs"]:
        print(
            f"线性粗估完整训练：{summary['estimated_total_seconds']:.2f} 秒｜"
            f"预计还需：{summary['estimated_remaining_seconds']:.2f} 秒"
        )
        print(f"预览 Accuracy：{summary['accuracy']:.1%}（只用于确认流程，不作为最终效果结论）")
    else:
        print(f"完整训练 Accuracy：{summary['accuracy']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="四个深度学习 Demo 的 CPU/GPU 速度教学入口")
    parser.add_argument("--lab", choices=tuple(LAB_DIRS), required=True, help="选择一个项目，避免一次启动全部模型")
    parser.add_argument("--devices", type=parse_devices, default=("cpu",), help="cpu、cuda:x 或 cpu,cuda:x")
    parser.add_argument("--target-epochs", type=int, default=None, help="完整教学目标；默认使用项目推荐值")
    parser.add_argument("--preview-epochs", type=int, default=None, help="只实跑少量 Epoch 并估算剩余时间")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target_epochs = args.target_epochs if args.target_epochs is not None else DEFAULT_EPOCHS[args.lab]
    actual_epochs = args.preview_epochs if args.preview_epochs is not None else target_epochs
    if target_epochs <= 0 or actual_epochs <= 0 or actual_epochs > target_epochs:
        parser.error("Epoch 必须大于 0，且 preview-epochs 不能超过 target-epochs。")
    if target_epochs > MAX_TARGET_EPOCHS[args.lab]:
        parser.error(f"{args.lab} 的速度演示 target-epochs 不能超过 {MAX_TARGET_EPOCHS[args.lab]}。")
    try:
        for device in args.devices:
            _cuda_index(device)
    except RuntimeError as error:
        parser.error(str(error))

    torch.set_num_threads(1)
    runner = _lab_runner(args.lab)
    print(f"项目：{args.lab.upper()}｜相同 seed={args.seed}｜目标 Epoch={target_epochs}｜实跑 Epoch={actual_epochs}")
    print("计时口径：训练引擎在 CUDA 前后同步；加速比不包含绘图。")
    summaries = [
        _run_once(runner, device, actual_epochs, target_epochs, args.seed)
        for device in args.devices
    ]
    for summary in summaries:
        _print_summary(summary)

    by_device = {str(summary["device"]).split(":", 1)[0]: summary for summary in summaries}
    if "cpu" in by_device and "cuda" in by_device:
        speedup = float(by_device["cpu"]["seconds_per_epoch"]) / float(by_device["cuda"]["seconds_per_epoch"])
        print(f"\n训练环节 GPU 加速比：{speedup:.2f}×（CPU 每 Epoch ÷ GPU 每 Epoch）")
        if speedup < 1:
            print("本次小模型 GPU 反而较慢：启动、数据传输和小算子开销超过了并行收益。")
        else:
            print("本次 GPU 更快；该倍数只对应当前模型、Batch、机器负载和软件环境。")
    print("估时按当前实跑 Epoch 线性外推；机器负载、首轮 GPU 热身和配置变化都会造成偏差。")


if __name__ == "__main__":
    main()
