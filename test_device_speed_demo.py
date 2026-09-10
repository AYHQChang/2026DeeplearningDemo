"""统一 CPU/GPU 速度入口的最小检查。"""

import argparse
from pathlib import Path
import subprocess
import sys

from device_speed_demo import parse_devices


def main() -> None:
    assert parse_devices("cpu") == ("cpu",)
    assert parse_devices("cpu,cuda:2") == ("cpu", "cuda:2")
    for invalid in ("auto", "cuda", "cpu,cpu", "cuda:0,cuda:1", "cpu,cuda:0,cuda:1"):
        try:
            parse_devices(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"无效设备组合应该被拒绝：{invalid}")

    script = Path(__file__).resolve().parent / "device_speed_demo.py"
    completed = subprocess.run(
        [
            sys.executable, "-B", "-X", "utf8", str(script), "--lab", "transformer",
            "--devices", "cpu", "--target-epochs", "12", "--preview-epochs", "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert "实跑 Epoch：1/12" in completed.stdout
    assert "预计还需" in completed.stdout
    assert "不作为最终效果结论" in completed.stdout
    rejected = subprocess.run(
        [
            sys.executable, "-B", "-X", "utf8", str(script), "--lab", "mlp",
            "--target-epochs", "0",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert rejected.returncode != 0 and "Epoch 必须大于 0" in rejected.stderr
    missing_gpu = subprocess.run(
        [
            sys.executable, "-B", "-X", "utf8", str(script), "--lab", "mlp",
            "--devices", "cpu,cuda:999", "--preview-epochs", "1",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert missing_gpu.returncode != 0 and "cuda:999 不可用" in missing_gpu.stderr
    assert "[cpu]" not in missing_gpu.stdout
    print("Device speed demo smoke test passed.")


if __name__ == "__main__":
    main()
