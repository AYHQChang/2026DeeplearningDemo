"""只读检查 Linux 多 GPU 服务器的 PCIe、驱动和 CUDA 初始化状态。

脚本不会安装软件、重启服务器、重载驱动或修改 GPU 设置。它依次检查：

1. ``lspci`` 是否看到预期数量的 NVIDIA 显示控制器，以及是否出现 ``rev ff``；
2. ``nvidia-smi`` 能识别多少张 GPU，并记录索引、PCI 地址、UUID 和名称；
3. 内核日志是否出现 ``RmInitAdapter failed``、``fallen off the bus`` 或 Xid；
4. PyTorch 在“全部 GPU 可见”和“每张 GPU 单独可见”时能否完成小张量计算。

默认按本课程服务器的 8 张卡检查，并重点关注 ``0000:4f:00.0``。其他机器可用
``--expected-gpus`` 和 ``--suspect-pci`` 修改这两个值。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_EXPECTED_GPUS = 8
DEFAULT_SUSPECT_PCI = "0000:4f:00.0"
GPU_CONTROLLER_WORDS = (
    "vga compatible controller",
    "3d controller",
    "display controller",
)
KERNEL_ERROR_PATTERNS = (
    re.compile(r"RmInitAdapter failed", re.IGNORECASE),
    re.compile(r"fallen off the bus", re.IGNORECASE),
    re.compile(r"NVRM:.*Xid", re.IGNORECASE),
    re.compile(r"PCIe Bus Error", re.IGNORECASE),
)
TORCH_MARKER = "__GPU_HEALTH_JSON__="


@dataclass
class CommandResult:
    """一次外部命令的可序列化结果。"""

    command: list[str]
    available: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.available and not self.timed_out and self.returncode == 0


@dataclass
class PciGpu:
    """由 lspci 识别的一张 NVIDIA 显示控制器。"""

    pci_bus_id: str
    description: str
    revision: str | None
    rev_ff: bool


@dataclass
class SmiGpu:
    """由 nvidia-smi 识别的一张 GPU。"""

    index: int
    pci_bus_id: str
    uuid: str
    name: str


@dataclass
class TorchProbe:
    """一个独立子进程中的最小 CUDA 测试结果。"""

    label: str
    visible_device: str | None
    status: str
    detail: dict[str, Any]
    stdout: str
    stderr: str
    returncode: int | None


def normalize_pci_bus_id(value: str) -> str:
    """把 4 位或 8 位 domain 的 PCI 地址统一为小写 4 位形式。"""

    match = re.search(
        r"(?:(?P<domain>[0-9a-fA-F]{4,8}):)?"
        r"(?P<bus>[0-9a-fA-F]{2}):(?P<slot>[0-9a-fA-F]{2})\."
        r"(?P<function>[0-7])",
        value,
    )
    if not match:
        return value.strip().lower()
    domain = (match.group("domain") or "0000")[-4:]
    return (
        f"{domain}:{match.group('bus')}:{match.group('slot')}."
        f"{match.group('function')}"
    ).lower()


def run_command(command: list[str], timeout: float = 10.0) -> CommandResult:
    """运行只读命令；命令不存在或超时也返回结果，不让主程序崩溃。"""

    if shutil.which(command[0]) is None:
        return CommandResult(command, False, None, "", "命令不存在")
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
        return CommandResult(
            command=command,
            available=True,
            returncode=completed.returncode,
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode("utf-8", "replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        return CommandResult(command, True, None, stdout.strip(), stderr.strip(), True)


def parse_lspci(output: str) -> list[PciGpu]:
    """提取 NVIDIA GPU 控制器，不把同一显卡的声卡功能算作 GPU。"""

    devices: list[PciGpu] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if "nvidia" not in lower or not any(word in lower for word in GPU_CONTROLLER_WORDS):
            continue
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        revision_match = re.search(r"\(rev\s+([0-9a-fA-F]{2})\)", line)
        revision = revision_match.group(1).lower() if revision_match else None
        devices.append(
            PciGpu(
                pci_bus_id=normalize_pci_bus_id(parts[0]),
                description=parts[1] if len(parts) > 1 else "",
                revision=revision,
                rev_ff=revision == "ff",
            )
        )
    return devices


def parse_nvidia_smi(output: str) -> tuple[list[SmiGpu], list[str]]:
    """解析 CSV 格式的 nvidia-smi 查询结果，并保留无法解析的行。"""

    devices: list[SmiGpu] = []
    malformed: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            malformed.append(line)
            continue
        try:
            index = int(fields[0])
        except ValueError:
            malformed.append(line)
            continue
        devices.append(
            SmiGpu(
                index=index,
                pci_bus_id=normalize_pci_bus_id(fields[1]),
                uuid=fields[2],
                name=fields[3],
            )
        )
    return devices, malformed


def collect_kernel_log(timeout: float) -> tuple[CommandResult, str]:
    """读取当前内核日志；普通账号无权读取 dmesg 时尝试 journalctl。"""

    dmesg = run_command(["dmesg", "--color=never"], timeout)
    if dmesg.ok:
        return dmesg, "dmesg"
    # CentOS 7 等旧系统的 dmesg 可能不支持 --color，因此再试一次最基础写法。
    plain_dmesg = run_command(["dmesg"], timeout)
    if plain_dmesg.ok:
        return plain_dmesg, "dmesg"
    journal = run_command(
        ["journalctl", "-k", "-b", "--no-pager", "--output=short-monotonic"],
        timeout,
    )
    if journal.ok and journal.stdout.strip():
        return journal, "journalctl"
    # 优先保留普通 dmesg 的权限错误，它通常比 journalctl 的空结果更直观。
    return plain_dmesg if plain_dmesg.available else dmesg, "dmesg"


def find_kernel_errors(output: str) -> list[str]:
    """筛选与 NVIDIA 初始化或 PCIe 总线有关的内核日志行。"""

    matches: list[str] = []
    seen: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if any(pattern.search(line) for pattern in KERNEL_ERROR_PATTERNS) and line not in seen:
            seen.add(line)
            matches.append(line)
    return matches


TORCH_PROBE_CODE = r"""
import json
import os

result = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
try:
    import torch
    result["torch_version"] = torch.__version__
    result["torch_cuda_runtime"] = torch.version.cuda
    result["cuda_available_before_init"] = torch.cuda.is_available()
    # 主动初始化，以便保留 CUDA/驱动返回的原始异常信息。
    torch.cuda.init()
    result["device_count"] = torch.cuda.device_count()
    torch.cuda.set_device(0)
    tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
    value = float((tensor * 2).sum().item())
    torch.cuda.synchronize()
    result["device_name"] = torch.cuda.get_device_name(0)
    result["calculation"] = value
    result["ok"] = value == 12.0
except Exception as error:
    result["ok"] = False
    result["error_type"] = type(error).__name__
    result["error"] = str(error)
print("__GPU_HEALTH_JSON__=" + json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if result.get("ok") else 1)
"""


def run_torch_probe(label: str, visible_device: str | None, timeout: float) -> TorchProbe:
    """在独立 Python 进程中执行一次 CUDA 初始化与小张量计算。"""

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if visible_device is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = visible_device
    try:
        completed = subprocess.run(
            [sys.executable, "-c", TORCH_PROBE_CODE],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode("utf-8", "replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        return TorchProbe(label, visible_device, "timeout", {}, stdout.strip(), stderr.strip(), None)

    detail: dict[str, Any] = {}
    for line in completed.stdout.splitlines():
        if line.startswith(TORCH_MARKER):
            try:
                detail = json.loads(line[len(TORCH_MARKER) :])
            except json.JSONDecodeError:
                detail = {"parse_error": line}
            break
    if detail.get("ok") is True and completed.returncode == 0:
        status = "ok"
    elif detail.get("error_type") == "ModuleNotFoundError" and "torch" in str(detail.get("error", "")):
        status = "torch_missing"
    else:
        status = "failed"
    return TorchProbe(
        label=label,
        visible_device=visible_device,
        status=status,
        detail=detail,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        returncode=completed.returncode,
    )


def print_command_problem(name: str, result: CommandResult) -> None:
    """用一行中文说明某个系统命令为何没有给出正常结果。"""

    if not result.available:
        print(f"[无法检查] {name}：系统中找不到命令。")
    elif result.timed_out:
        print(f"[异常] {name}：命令执行超时。")
    else:
        detail = result.stderr or result.stdout or "没有输出"
        print(f"[异常] {name}：退出码 {result.returncode}；{detail[:500]}")


def print_torch_probe(probe: TorchProbe) -> None:
    """打印一条 CUDA 子进程检查结果。"""

    if probe.status == "ok":
        name = probe.detail.get("device_name", "未知型号")
        count = probe.detail.get("device_count", "?")
        print(f"[OK] {probe.label}：CUDA 初始化和张量计算成功；可见 {count} 张卡；{name}")
        return
    if probe.status == "timeout":
        print(f"[异常] {probe.label}：CUDA 子进程超时，可能卡在驱动初始化。")
        return
    error = probe.detail.get("error") or probe.stderr or probe.stdout or "没有返回错误详情"
    print(f"[异常] {probe.label}：{error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读检查多 GPU Linux 服务器的 PCIe、NVIDIA 驱动和 CUDA 初始化状态。",
    )
    parser.add_argument(
        "--expected-gpus",
        type=int,
        default=DEFAULT_EXPECTED_GPUS,
        help=f"预期物理 GPU 数量，默认 {DEFAULT_EXPECTED_GPUS}。",
    )
    parser.add_argument(
        "--suspect-pci",
        default=DEFAULT_SUSPECT_PCI,
        help=f"重点检查的 PCI 地址，默认 {DEFAULT_SUSPECT_PCI}；传入空字符串可关闭。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="每条系统命令或 CUDA 子进程的超时秒数，默认 20。",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="只查 PCI、驱动和内核日志，不启动 PyTorch CUDA 测试。",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="可选：把完整结构化报告保存到指定 JSON 文件。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_gpus < 1:
        raise SystemExit("--expected-gpus 必须大于 0。")
    if args.timeout <= 0:
        raise SystemExit("--timeout 必须大于 0。")

    suspect_pci = normalize_pci_bus_id(args.suspect_pci) if args.suspect_pci.strip() else ""
    report: dict[str, Any] = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "unknown"),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "expected_gpus": args.expected_gpus,
        "suspect_pci": suspect_pci or None,
        "original_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    anomalies: list[str] = []
    limitations: list[str] = []

    print("=" * 72)
    print("Linux 多 GPU 服务器只读健康检查")
    print("=" * 72)
    print(f"主机：{report['hostname']}  Python：{report['python']}")
    print(f"预期物理 GPU：{args.expected_gpus} 张  重点 PCI：{suspect_pci or '未指定'}")
    original_visibility = report["original_cuda_visible_devices"]
    if original_visibility is not None:
        print(
            "[提醒] 当前终端设置了 CUDA_VISIBLE_DEVICES="
            f"{original_visibility}。脚本不会越过这项课堂分配；PyTorch 只测试当前可见范围，"
            "并跳过全服务器逐卡测试。教师做整机检查时应在未设置该变量的管理终端中运行。"
        )

    print("\n[1/4] PCIe 物理设备")
    lspci_result = run_command(["lspci", "-Dnn"], args.timeout)
    pci_gpus = parse_lspci(lspci_result.stdout) if lspci_result.ok else []
    report["lspci"] = asdict(lspci_result)
    report["pci_gpus"] = [asdict(gpu) for gpu in pci_gpus]
    if not lspci_result.ok:
        print_command_problem("lspci", lspci_result)
        limitations.append("无法用 lspci 核对物理 GPU 和 rev ff。")
    else:
        print(f"lspci 识别到 {len(pci_gpus)} 个 NVIDIA GPU 控制器。")
        for gpu in pci_gpus:
            revision = gpu.revision or "未知"
            marker = "异常：rev ff" if gpu.rev_ff else "OK"
            print(f"  [{marker}] {gpu.pci_bus_id}  rev {revision}  {gpu.description}")
        if len(pci_gpus) != args.expected_gpus:
            anomalies.append(
                f"lspci 识别到 {len(pci_gpus)} 张 GPU，与预期 {args.expected_gpus} 张不一致。"
            )
        rev_ff_devices = [gpu.pci_bus_id for gpu in pci_gpus if gpu.rev_ff]
        if rev_ff_devices:
            anomalies.append("lspci 出现 rev ff：" + ", ".join(rev_ff_devices))
        if suspect_pci:
            suspect = next((gpu for gpu in pci_gpus if gpu.pci_bus_id == suspect_pci), None)
            if suspect is None:
                anomalies.append(f"lspci 未找到重点设备 {suspect_pci}。")
            elif suspect.rev_ff:
                anomalies.append(f"重点设备 {suspect_pci} 返回 rev ff，PCI 配置空间不可正常读取。")

    print("\n[2/4] NVIDIA 驱动可见设备")
    smi_command = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,uuid,name",
        "--format=csv,noheader,nounits",
    ]
    smi_result = run_command(smi_command, args.timeout)
    # 某些驱动错误会让命令以非零状态退出，但 stdout 仍可能包含其余健康 GPU。
    # 因此无论退出码如何都尝试解析，便于继续用 UUID 隔离测试可见设备。
    smi_gpus, malformed_smi = parse_nvidia_smi(smi_result.stdout)
    report["nvidia_smi"] = asdict(smi_result)
    report["smi_gpus"] = [asdict(gpu) for gpu in smi_gpus]
    report["nvidia_smi_malformed_lines"] = malformed_smi
    if not smi_result.ok:
        print_command_problem("nvidia-smi", smi_result)
        anomalies.append("nvidia-smi 无法完成 GPU 列表查询。")
        if smi_gpus:
            print(f"  仍从部分输出中解析到 {len(smi_gpus)} 张 GPU，将继续做隔离测试：")
            for gpu in smi_gpus:
                print(f"  [部分结果] index={gpu.index}  {gpu.pci_bus_id}  {gpu.uuid}  {gpu.name}")
    else:
        print(f"nvidia-smi 识别到 {len(smi_gpus)} 张 GPU。")
        for gpu in smi_gpus:
            print(f"  [OK] index={gpu.index}  {gpu.pci_bus_id}  {gpu.uuid}  {gpu.name}")
    if malformed_smi:
        anomalies.append(f"nvidia-smi 有 {len(malformed_smi)} 行无法解析。")
    if len(smi_gpus) != args.expected_gpus:
        anomalies.append(
            f"nvidia-smi 只识别到 {len(smi_gpus)} 张 GPU，预期为 {args.expected_gpus} 张。"
        )
    if suspect_pci and all(gpu.pci_bus_id != suspect_pci for gpu in smi_gpus):
        anomalies.append(f"nvidia-smi 的 GPU 列表中没有重点设备 {suspect_pci}。")

    if suspect_pci and shutil.which("nvidia-smi") is not None:
        suspect_gpu = next((gpu for gpu in smi_gpus if gpu.pci_bus_id == suspect_pci), None)
        # 已被驱动识别时优先使用 UUID；未识别时只能按 PCI 地址直接询问驱动。
        suspect_selector = suspect_gpu.uuid if suspect_gpu else suspect_pci
        suspect_result = run_command(["nvidia-smi", "-i", suspect_selector, "-q"], args.timeout)
        report["suspect_nvidia_smi"] = asdict(suspect_result)
        if suspect_result.ok:
            print(f"[OK] nvidia-smi 可以单独查询重点设备 {suspect_pci}。")
        else:
            detail = suspect_result.stderr or suspect_result.stdout or "没有错误详情"
            print(f"[异常] nvidia-smi 无法单独查询 {suspect_pci}：{detail[:500]}")
            anomalies.append(f"nvidia-smi 无法单独查询重点设备 {suspect_pci}。")

    print("\n[3/4] 当前启动的内核日志")
    kernel_result, kernel_source = collect_kernel_log(args.timeout)
    kernel_errors = find_kernel_errors(kernel_result.stdout) if kernel_result.ok else []
    report["kernel_log_source"] = kernel_source
    # JSON 只保留匹配行和命令状态，不复制可能很大的完整内核日志。
    report["kernel_log_command"] = {
        "command": kernel_result.command,
        "available": kernel_result.available,
        "returncode": kernel_result.returncode,
        "stderr": kernel_result.stderr,
        "timed_out": kernel_result.timed_out,
        "stdout_line_count": len(kernel_result.stdout.splitlines()),
    }
    report["kernel_errors"] = kernel_errors
    if not kernel_result.ok:
        print_command_problem(kernel_source, kernel_result)
        print("  普通账号可能没有读取内核日志的权限；这不等于内核中没有错误。")
        limitations.append("当前账号无法读取内核日志，不能排除 RmInitAdapter/Xid/掉总线错误。")
    elif kernel_errors:
        print(f"找到 {len(kernel_errors)} 条相关日志：")
        for line in kernel_errors[-30:]:
            print(f"  [异常] {line}")
        if len(kernel_errors) > 30:
            print(f"  仅显示最后 30 条；完整匹配共 {len(kernel_errors)} 条。")
        anomalies.append("内核日志出现 NVIDIA 初始化、Xid 或 PCIe 总线错误。")
    else:
        print(f"[OK] {kernel_source} 中未匹配到本脚本关注的 GPU/PCIe 错误关键词。")

    print("\n[4/4] PyTorch CUDA 独立子进程")
    torch_probes: list[TorchProbe] = []
    if args.skip_torch:
        print("已通过 --skip-torch 跳过。")
        limitations.append("未执行 PyTorch CUDA 初始化测试。")
    else:
        global_label = "当前终端可见范围" if original_visibility is not None else "全部 GPU 可见"
        global_probe = run_torch_probe(global_label, original_visibility, args.timeout)
        torch_probes.append(global_probe)
        print_torch_probe(global_probe)

        if original_visibility is not None:
            print("检测到课堂 GPU 可见范围限制，因此不尝试其他未分配 GPU。")
            limitations.append("受 CUDA_VISIBLE_DEVICES 限制，未执行全服务器逐卡 CUDA 测试。")
        elif global_probe.status == "torch_missing":
            print("当前 Python 环境没有安装 PyTorch，跳过重复的逐卡导入测试。")
        elif smi_gpus:
            for gpu in smi_gpus:
                # UUID 不受 GPU 索引重新编号影响，比 CUDA_VISIBLE_DEVICES=数字更可靠。
                label = f"单卡 index={gpu.index} / {gpu.pci_bus_id}"
                probe = run_torch_probe(label, gpu.uuid, args.timeout)
                torch_probes.append(probe)
                print_torch_probe(probe)
        elif shutil.which("nvidia-smi") is not None:
            print("nvidia-smi 没有返回可用 UUID，改用数字索引逐个尝试；索引与物理卡可能无法对应。")
            for index in range(args.expected_gpus):
                probe = run_torch_probe(f"候选索引 {index}", str(index), args.timeout)
                torch_probes.append(probe)
                print_torch_probe(probe)
        else:
            limitations.append("没有 nvidia-smi 设备清单，未执行逐卡 CUDA 测试。")

        if global_probe.status == "torch_missing":
            limitations.append("当前 Python 环境没有安装 PyTorch，无法检查 CUDA 初始化。")
        elif global_probe.status != "ok":
            anomalies.append(f"{global_label}下，PyTorch CUDA 初始化或张量计算失败。")
        failed_single = [probe.label for probe in torch_probes[1:] if probe.status not in {"ok", "torch_missing"}]
        if failed_single:
            anomalies.append("以下单卡 CUDA 测试失败：" + ", ".join(failed_single))

    report["torch_probes"] = [asdict(probe) for probe in torch_probes]
    report["anomalies"] = anomalies
    report["limitations"] = limitations

    print("\n" + "=" * 72)
    print("诊断结论")
    print("=" * 72)
    if anomalies:
        print(f"[发现异常] 本机不满足“{args.expected_gpus} 张卡均可正常初始化”的条件：")
        for item in anomalies:
            print(f"  - {item}")
        if any("rev ff" in item for item in anomalies) and kernel_errors:
            print(
                "\n综合证据同时包含 PCI rev ff 与内核驱动/总线错误，符合 GPU 掉总线或"
                "设备无法初始化的表现。脚本只负责定位，不会自动修复。"
            )
    elif limitations:
        print("[未发现明确异常，但检查不完整]")
    else:
        print(f"[通过] PCI、驱动和 CUDA 检查均通过，识别到预期的 {args.expected_gpus} 张 GPU。")

    if limitations:
        print("\n检查限制：")
        for item in limitations:
            print(f"  - {item}")
    print("\n请把完整终端输出交给服务器管理员；不要自行重载驱动、热移除设备或重启服务器。")

    if args.json:
        output_path = args.json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 报告已保存：{output_path}")

    if anomalies:
        return 1
    if limitations:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
