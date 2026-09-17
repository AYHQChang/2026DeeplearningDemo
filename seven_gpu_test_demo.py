#!/usr/bin/env python3
"""验证服务器上剩余七张 GPU 能否共同、逐张完成一次真实的 PyTorch 训练。

本脚本只读取 GPU 状态并运行很小的 MLP，不会修改驱动、重启服务器或隐藏设备。
默认场景是：一张故障卡已经无法被 nvidia-smi 识别，nvidia-smi 还能返回七张好卡。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any


RESULT_MARKER = "__SEVEN_GPU_TEST_JSON__="
DEFAULT_SUSPECT_PCI = "0000:4f:00.0"


@dataclass(frozen=True)
class GpuInfo:
    """nvidia-smi 返回的一张 GPU。"""

    physical_index: int
    pci_bus_id: str
    uuid: str
    name: str


@dataclass
class ProbeResult:
    """一个独立 Python 子进程的测试结果。"""

    label: str
    visible_devices: str
    status: str
    returncode: int | None
    elapsed_seconds: float
    detail: dict[str, Any]
    stdout: str
    stderr: str


CHILD_TEST_CODE = r"""
import json
import math
import os
import sys
import traceback

marker = "__SEVEN_GPU_TEST_JSON__="
expected_count = int(sys.argv[1])
steps = int(sys.argv[2])
result = {
    "ok": False,
    "expected_count": expected_count,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "devices": [],
}

try:
    import torch
    from torch import nn

    result["torch_version"] = torch.__version__
    result["torch_cuda_runtime"] = torch.version.cuda
    result["cuda_available_before_init"] = torch.cuda.is_available()

    # 主动初始化 CUDA。若驱动仍被故障卡拖累，这里会保留原始异常，而不是只报告 False。
    torch.cuda.init()
    actual_count = torch.cuda.device_count()
    result["actual_count"] = actual_count
    if actual_count != expected_count:
        raise RuntimeError(
            f"CUDA 初始化后检测到 {actual_count} 张卡，预期为 {expected_count} 张"
        )

    for logical_index in range(actual_count):
        record = {"logical_index": logical_index, "ok": False}
        try:
            device = torch.device(f"cuda:{logical_index}")
            torch.cuda.set_device(device)
            torch.cuda.manual_seed(2026 + logical_index)

            properties = torch.cuda.get_device_properties(device)
            model = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
            ).to(device)
            inputs = torch.randn(64, 8, device=device)
            targets = torch.randint(0, 2, (64,), device=device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            criterion = nn.CrossEntropyLoss()
            first_parameter = next(model.parameters())
            before_update = first_parameter.detach().clone()

            losses = []
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))

            torch.cuda.synchronize(device)
            parameter_delta = float(
                (first_parameter.detach() - before_update).abs().max().item()
            )
            finite_losses = all(math.isfinite(value) for value in losses)
            record.update(
                {
                    "name": torch.cuda.get_device_name(device),
                    "total_memory_mib": round(properties.total_memory / 1024**2),
                    "first_loss": losses[0],
                    "final_loss": losses[-1],
                    "parameter_max_change": parameter_delta,
                    "ok": finite_losses and parameter_delta > 0.0,
                }
            )
            if not record["ok"]:
                record["error"] = "损失不是有限数，或参数在训练后没有发生变化"
        except Exception as error:
            record.update(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        result["devices"].append(record)

    result["ok"] = len(result["devices"]) == expected_count and all(
        item.get("ok") is True for item in result["devices"]
    )
except Exception as error:
    result.update(
        {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    )

print(marker + json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if result.get("ok") else 1)
"""


def normalize_pci_bus_id(value: str) -> str:
    """把 PCI 地址统一成 0000:4f:00.0 形式，便于比较。"""

    normalized = value.strip().lower()
    if normalized.count(":") == 1:
        normalized = f"0000:{normalized}"
    parts = normalized.split(":")
    if len(parts) != 3 or "." not in parts[2]:
        return normalized
    device, function = parts[2].split(".", maxsplit=1)
    try:
        # nvidia-smi 常把 domain 打印成 8 位，例如 00000000:4f:00.0；
        # lspci 通常打印成 4 位。这里统一为标准的 4:2:2.1 形式。
        return (
            f"{int(parts[0], 16):04x}:{int(parts[1], 16):02x}:"
            f"{int(device, 16):02x}.{int(function, 16):x}"
        )
    except ValueError:
        return normalized


def query_driver_visible_gpus(timeout: int) -> tuple[list[GpuInfo], str]:
    """用 nvidia-smi 获取驱动当前真正能够识别的 GPU。"""

    command = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,uuid,name",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise RuntimeError("找不到 nvidia-smi，请确认已在 Linux GPU 服务器上运行") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"nvidia-smi 超过 {timeout} 秒仍未返回") from error

    raw_output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi 执行失败（退出码 {completed.returncode}）：\n{raw_output}"
        )

    gpus: list[GpuInfo] = []
    for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            raise RuntimeError(f"无法解析 nvidia-smi 第 {line_number} 行：{line}")
        try:
            physical_index = int(fields[0])
        except ValueError as error:
            raise RuntimeError(f"GPU 编号不是整数：{fields[0]}") from error
        gpus.append(
            GpuInfo(
                physical_index=physical_index,
                pci_bus_id=normalize_pci_bus_id(fields[1]),
                uuid=fields[2],
                name=fields[3],
            )
        )
    return gpus, raw_output


def run_pytorch_probe(
    *,
    label: str,
    visible_devices: str,
    expected_count: int,
    steps: int,
    timeout: int,
) -> ProbeResult:
    """在独立进程中初始化指定 GPU，并完成一次最小 MLP 训练。"""

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", CHILD_TEST_CODE, str(expected_count), str(steps)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        elapsed = time.monotonic() - started
        stdout = error.stdout.decode("utf-8", "replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        return ProbeResult(
            label=label,
            visible_devices=visible_devices,
            status="timeout",
            returncode=None,
            elapsed_seconds=round(elapsed, 3),
            detail={"error": f"测试超过 {timeout} 秒"},
            stdout=stdout.strip(),
            stderr=stderr.strip(),
        )

    detail: dict[str, Any] = {}
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            try:
                detail = json.loads(line[len(RESULT_MARKER) :])
            except json.JSONDecodeError:
                detail = {"error": "子进程结果 JSON 解析失败", "raw_line": line}
            break

    if detail.get("ok") is True and completed.returncode == 0:
        status = "pass"
    elif not detail:
        status = "no_result"
        detail = {"error": "子进程没有返回结构化测试结果"}
    else:
        status = "fail"

    return ProbeResult(
        label=label,
        visible_devices=visible_devices,
        status=status,
        returncode=completed.returncode,
        elapsed_seconds=round(time.monotonic() - started, 3),
        detail=detail,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def concise_error(probe: ProbeResult) -> str:
    """提取最有用的错误，同时在 JSON 报告中保留完整 traceback。"""

    detail = probe.detail
    error = str(detail.get("error", "未知错误"))
    error_type = detail.get("error_type")
    if error_type:
        return f"{error_type}: {error}"
    for device in detail.get("devices", []):
        if device.get("ok") is not True:
            device_error = str(device.get("error", "未知设备错误"))
            device_error_type = device.get("error_type")
            prefix = f"cuda:{device.get('logical_index', '?')}"
            if device_error_type:
                return f"{prefix} {device_error_type}: {device_error}"
            return f"{prefix}: {device_error}"
    return error


def print_discovery(gpus: list[GpuInfo]) -> None:
    print("\n驱动当前可见的 GPU：")
    print("物理编号  PCI 地址          UUID                              型号")
    print("--------  ----------------  --------------------------------  ------------------------------")
    for gpu in gpus:
        print(
            f"{gpu.physical_index:<8}  {gpu.pci_bus_id:<16}  "
            f"{gpu.uuid:<32}  {gpu.name}"
        )


def print_probe_details(probe: ProbeResult) -> None:
    print(f"\n[{probe.status.upper()}] {probe.label}（{probe.elapsed_seconds:.2f} 秒）")
    if probe.status != "pass":
        print(f"  错误：{concise_error(probe)}")
        if probe.stderr:
            print("  stderr：")
            for line in probe.stderr.splitlines():
                print(f"    {line}")
        return
    for device in probe.detail.get("devices", []):
        print(
            "  "
            f"cuda:{device['logical_index']} | {device['name']} | "
            f"loss {device['first_loss']:.6f} -> {device['final_loss']:.6f} | "
            f"参数最大变化 {device['parameter_max_change']:.3e}"
        )


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整 JSON 报告已保存：{path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证 nvidia-smi 可见的七张 GPU 能否共同、逐张完成 PyTorch MLP 训练。"
    )
    parser.add_argument(
        "--expected-gpus",
        type=int,
        default=7,
        help="预期 nvidia-smi 返回的 GPU 数量，默认 7",
    )
    parser.add_argument(
        "--suspect-pci",
        default=DEFAULT_SUSPECT_PCI,
        help=f"不得进入测试的故障卡 PCI 地址，默认 {DEFAULT_SUSPECT_PCI}",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="每张卡执行的最小 MLP 训练步数，默认 3",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="每个子测试的超时时间（秒），默认 60",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="可选：把完整结果写入指定 JSON 文件，例如 logs/seven_gpu_test_report.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.expected_gpus < 1:
        print("[ERROR] --expected-gpus 必须大于 0。", file=sys.stderr)
        return 2
    if args.steps < 1:
        print("[ERROR] --steps 必须大于 0。", file=sys.stderr)
        return 2
    if args.timeout < 1:
        print("[ERROR] --timeout 必须大于 0。", file=sys.stderr)
        return 2

    # 这是整机验收脚本。若继承了课堂分卡变量，主动停止，避免把局部结果误判为整机结果。
    inherited_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited_mask is not None:
        print(
            "[ERROR] 当前终端已经设置 CUDA_VISIBLE_DEVICES="
            f"{inherited_mask!r}。\n"
            "请在未分配 GPU 的管理终端运行本脚本；脚本会自行使用七张卡的 UUID。",
            file=sys.stderr,
        )
        return 2

    print("=" * 76)
    print(f"{args.expected_gpus} 张 GPU 课程可用性验收 Demo")
    print("测试内容：多卡共同初始化 + 每张卡独立初始化 + 最小 MLP 前向/反向/参数更新")
    print("=" * 76)

    try:
        gpus, smi_output = query_driver_visible_gpus(args.timeout)
    except RuntimeError as error:
        print(f"\n[FAIL] {error}", file=sys.stderr)
        return 1

    print_discovery(gpus)
    suspect_pci = normalize_pci_bus_id(args.suspect_pci) if args.suspect_pci else ""
    discovery_errors: list[str] = []
    if len(gpus) != args.expected_gpus:
        discovery_errors.append(
            f"nvidia-smi 返回 {len(gpus)} 张卡，预期为 {args.expected_gpus} 张"
        )
    if suspect_pci and any(gpu.pci_bus_id == suspect_pci for gpu in gpus):
        discovery_errors.append(f"故障卡 {suspect_pci} 仍出现在 nvidia-smi 结果中")
    if len({gpu.uuid for gpu in gpus}) != len(gpus):
        discovery_errors.append("nvidia-smi 返回了重复的 GPU UUID")

    if discovery_errors:
        print("\n[FAIL] GPU 候选列表不满足安全测试条件：", file=sys.stderr)
        for message in discovery_errors:
            print(f"  - {message}", file=sys.stderr)
        print("未启动 PyTorch，避免测试到数量或身份不明确的设备。", file=sys.stderr)
        report = {
            "ok": False,
            "phase": "discovery",
            "errors": discovery_errors,
            "nvidia_smi_output": smi_output,
            "gpus": [asdict(gpu) for gpu in gpus],
        }
        if args.json:
            write_json_report(args.json, report)
        return 1

    all_gpu_uuids = ",".join(gpu.uuid for gpu in gpus)
    print(f"\n第 1 阶段：把 {args.expected_gpus} 个 UUID 一起交给一个 PyTorch 进程。")
    combined_probe = run_pytorch_probe(
        label=f"七卡组合测试（预期 {args.expected_gpus} 张）",
        visible_devices=all_gpu_uuids,
        expected_count=args.expected_gpus,
        steps=args.steps,
        timeout=args.timeout,
    )
    print_probe_details(combined_probe)

    print("\n第 2 阶段：每个 UUID 分别进入独立进程，排除单卡初始化问题。")
    individual_probes: list[ProbeResult] = []
    for ordinal, gpu in enumerate(gpus, start=1):
        print(
            f"  正在测试 {ordinal}/{len(gpus)}：物理 GPU {gpu.physical_index} "
            f"({gpu.pci_bus_id}) ...",
            flush=True,
        )
        probe = run_pytorch_probe(
            label=f"物理 GPU {gpu.physical_index} / {gpu.pci_bus_id}",
            visible_devices=gpu.uuid,
            expected_count=1,
            steps=args.steps,
            timeout=args.timeout,
        )
        individual_probes.append(probe)

    print("\n逐卡结果：")
    print("物理编号  PCI 地址          结果   说明")
    print("--------  ----------------  -----  ------------------------------------------")
    for gpu, probe in zip(gpus, individual_probes):
        if probe.status == "pass":
            device = probe.detail["devices"][0]
            explanation = (
                f"{device['name']}；loss {device['first_loss']:.4f} -> "
                f"{device['final_loss']:.4f}"
            )
        else:
            explanation = concise_error(probe).replace("\n", " ")
        print(
            f"{gpu.physical_index:<8}  {gpu.pci_bus_id:<16}  "
            f"{probe.status.upper():<5}  {explanation}"
        )

    success = combined_probe.status == "pass" and all(
        probe.status == "pass" for probe in individual_probes
    )
    report = {
        "ok": success,
        "expected_gpus": args.expected_gpus,
        "suspect_pci": suspect_pci,
        "nvidia_smi_output": smi_output,
        "gpus": [asdict(gpu) for gpu in gpus],
        "combined_probe": asdict(combined_probe),
        "individual_probes": [asdict(probe) for probe in individual_probes],
    }
    if args.json:
        write_json_report(args.json, report)

    print("\n" + "=" * 76)
    if success:
        print(
            f"[PASS] {args.expected_gpus} 张卡共同测试和 {args.expected_gpus} 次独立测试全部通过。\n"
            f"结论：从当前账号、当前 Conda 环境和当前驱动路径看，这 {args.expected_gpus} 张卡可以用于课程实验。"
        )
        return 0

    print(
        "[FAIL] 至少一个测试未通过。不要仅凭 nvidia-smi 可见就把七张卡分给学生。\n"
        "请把终端完整输出或 --json 生成的报告交给管理员继续定位。"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
