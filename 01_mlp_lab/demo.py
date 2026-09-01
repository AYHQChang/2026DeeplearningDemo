"""MLP 实验室的 VS Code / PowerShell 一键运行入口。

这个文件只负责四件事：

1. 用 ``argparse`` 接收设备、数据、组件、随机种子和运行模式；
2. 把命令行参数转换成 ``ExperimentConfig``，调用 ``mlp_lab`` 的共享训练实现；
3. 在终端输出结果表，并显示或保存阶段图、诊断图和组件对比矩阵。
4. 默认保存结构化 JSON、终端摘要 TXT，并把关键指标追加到 CSV 历史表。

模型定义、数据划分、损失计算和训练循环都不在这里复制。这样 Notebook 与脚本
始终执行完全相同的算法。脚本不联网，所有数据由本地函数生成。

常用命令：

    python demo.py --device cpu --mode fast --experiment baseline
    python demo.py --experiment activation --dataset spiral
    python demo.py --experiment optimizer --save-dir outputs --no-show

``baseline`` 输出“训练阶段故事图”和“四联诊断图”；组件实验输出一个按列组织的
对比矩阵。``--no-show`` 用于自动验收，避免弹出图形窗口。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

from core import (
    ExperimentConfig,
    TrainingResult,
    base_config,
    format_result_table,
    make_dataset,
    plot_comparison,
    plot_dataset_gallery,
    plot_final_diagnosis,
    plot_training_story,
    print_result_table,
    resolve_device,
    run_comparison,
    save_or_show,
    train_experiment,
)


LAB_DIR = Path(__file__).resolve().parent


# 组件标识用于命令行参数；中文标题直接说明这组图要回答的问题。
COMPARISON_TITLES = {
    "activation": "激活函数对比：非线性形状如何影响学习",
    "depth": "网络深度对比：更深是否一定更好",
    "loss": "损失函数对比：分类目标应该怎样表达",
    "optimizer": "优化器对比：参数怎样走向低损失区域",
    "dropout": "Dropout 对比：正则化与欠拟合之间的平衡",
    "width": "隐藏层宽度对比：容量、速度与泛化",
}


# CSV 只保留便于筛选和排序的关键字段；完整逐轮曲线保存在每次运行的 JSON 中。
HISTORY_FIELDS = (
    "recorded_at",
    "run_id",
    "experiment",
    "source_core_sha256",
    "config_name",
    "dataset",
    "seed",
    "noise",
    "n_samples",
    "epochs",
    "hidden_sizes",
    "activation",
    "loss_name",
    "optimizer",
    "learning_rate",
    "dropout",
    "resolved_device",
    "final_train_accuracy",
    "final_test_accuracy",
    "final_train_loss",
    "final_gradient_norm",
    "parameter_count",
    "elapsed_seconds",
)


# =============================================================================
# 1. 命令行参数解析
# =============================================================================


def parse_hidden_sizes(text: str) -> tuple[int, ...]:
    """解析命令行中的隐藏层宽度列表。

    Args:
        text: 逗号分隔的正整数，例如 ``"32,32"`` 或 ``"64"``。

    Returns:
        整数元组，例如 ``(32, 32)``。元组长度就是隐藏层数量。

    Raises:
        argparse.ArgumentTypeError: 含非整数、空列表或非正数。使用 argparse
            专用异常后，命令行会显示简洁错误和帮助信息，而不是完整堆栈。
    """

    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("hidden-sizes 应写成 32,32 这样的整数列表。") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("hidden-sizes 至少包含一个正整数。")
    return values


def build_parser() -> argparse.ArgumentParser:
    """创建包含全部公开命令行接口的参数解析器。

    Returns:
        配置完成的 ``ArgumentParser``。函数只定义接口，不读取 ``sys.argv``，
        因此可以单独测试帮助文本和参数类型。
    """

    parser = argparse.ArgumentParser(description="MLP 组件与决策边界课堂实验室")
    parser.add_argument("--device", default="cpu", help="运行设备：cpu、auto、cuda 或 cuda:编号；课堂默认 CPU")
    parser.add_argument("--mode", choices=("fast", "extended"), default="fast", help="课堂快速模式或课后扩展模式")
    parser.add_argument("--dataset", choices=("moons", "circles", "spiral"), default="spiral", help="二维数据集")
    parser.add_argument(
        "--experiment",
        choices=("gallery", "baseline", *COMPARISON_TITLES, "all"),
        default="baseline",
        help="要运行的基线或单变量对照",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 mode 的训练轮数")
    parser.add_argument("--noise", type=float, default=None, help="覆盖数据集的默认噪声")
    parser.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=(32, 32), help="隐藏层宽度，例如 32,32")
    parser.add_argument("--activation", choices=("relu", "tanh", "sigmoid"), default="relu")
    parser.add_argument("--loss", dest="loss_name", choices=("cross_entropy", "mse"), default="cross_entropy")
    parser.add_argument("--optimizer", choices=("sgd", "momentum", "adam"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--save-dir", type=Path, default=None, help="可选：把课堂图保存到此目录")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LAB_DIR / "logs",
        help="实验日志目录，默认固定为 01_mlp_lab/logs",
    )
    parser.add_argument("--no-log", action="store_true", help="本次不写 JSON、TXT 和 CSV 实验日志")
    parser.add_argument("--no-show", action="store_true", help="不弹出绘图窗口，适合自动化测试")
    return parser


def make_base_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """把命令行 Namespace 合并到 fast/extended 基线配置。

    Args:
        args: ``build_parser().parse_args()`` 返回的命令行参数。

    Returns:
        不可变 ``ExperimentConfig``。``--epochs`` 未提供时保留 mode 默认值；
        其他公开组件参数覆盖基线对应字段。
    """

    config = base_config(dataset=args.dataset, mode=args.mode, seed=args.seed, device=args.device)
    updates = {
        "hidden_sizes": args.hidden_sizes,
        "activation": args.activation,
        "loss_name": args.loss_name,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
    }
    if args.epochs is not None:
        updates["epochs"] = args.epochs
    if args.noise is not None:
        updates["noise"] = args.noise
    config = replace(config, **updates)
    hidden = "x".join(map(str, config.hidden_sizes))
    loss_label = {"cross_entropy": "CE", "mse": "MSE"}[config.loss_name]
    name = f"配置：{hidden}｜{config.activation}｜{loss_label}｜{config.optimizer}"
    return replace(config, name=name)


# =============================================================================
# 2. 可比较实验日志
# =============================================================================


def _sha256_file(path: Path) -> str:
    """计算源码文件的 SHA-256，用于识别基线实现是否发生变化。

    Args:
        path: 需要读取的本地源码文件。

    Returns:
        64 位十六进制摘要。即使配置相同，只要核心实现改变，日志中也能看出
        两次结果来自不同代码版本。
    """

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _result_record(result: TrainingResult) -> dict[str, object]:
    """把训练结果转换成可以直接写入 JSON 的字典。

    Args:
        result: 一次完整训练产生的 ``TrainingResult``。

    Returns:
        包含配置、数据规模、最终指标和四条逐 epoch 曲线的普通字典。模型参数
        与快照体积较大，不写入日志；需要保存模型时应使用 ``torch.save``。
    """

    return {
        "config": asdict(result.config),
        "data": {
            "train_samples": len(result.data.x_train),
            "test_samples": len(result.data.x_test),
            "n_classes": result.data.n_classes,
        },
        "metrics": {
            "final_train_accuracy": result.train_accuracy[-1],
            "final_test_accuracy": result.final_test_accuracy,
            "final_train_loss": result.train_loss[-1],
            "minimum_train_loss": min(result.train_loss),
            "final_gradient_norm": result.gradient_norm[-1],
            "parameter_count": result.parameter_count,
            "elapsed_seconds": result.elapsed_seconds,
        },
        "history": {
            "train_loss": result.train_loss,
            "train_accuracy": result.train_accuracy,
            "test_accuracy": result.test_accuracy,
            "gradient_norm": result.gradient_norm,
        },
    }


def save_experiment_log(
    results: list[TrainingResult],
    experiment: str,
    requested_device: str,
    resolved_device: str,
    mode: str,
    log_dir: Path,
) -> tuple[Path, Path, Path]:
    """保存一次训练任务的 JSON、TXT，并追加 CSV 历史记录。

    Args:
        results: 基线的一项结果，或某个组件对比的多项结果。
        experiment: ``baseline`` 或组件名称，用于文件名和筛选。
        requested_device: 命令行传入的 ``auto``、``cpu`` 或 ``cuda``。
        resolved_device: 实际运行的 ``cpu`` 或 ``cuda``。
        mode: ``fast`` 或 ``extended``。
        log_dir: 日志目录；不存在时自动创建。

    Returns:
        ``(json_path, text_path, csv_path)``。JSON 保留完整曲线，TXT 与终端表格
        一致，CSV 每个配置占一行，适合按日期或基线参数进行快速比较。

    Raises:
        ValueError: ``results`` 为空。没有训练的 ``gallery`` 不调用本函数。
    """

    if not results:
        raise ValueError("results 不能为空。")
    log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    run_id = now.strftime("%Y%m%d_%H%M%S_%f")
    mlp_sources = [LAB_DIR / "core.py", *sorted((LAB_DIR / "mlp_lab").glob("*.py"))]
    digest = hashlib.sha256()
    for source_path in mlp_sources:
        digest.update(source_path.name.encode("utf-8"))
        digest.update(source_path.read_bytes())
    source_hashes = {
        "mlp_lab": digest.hexdigest(),
        "demo.py": _sha256_file(LAB_DIR / "demo.py"),
    }
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "recorded_at": now.isoformat(timespec="seconds"),
        "experiment": experiment,
        "mode": mode,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "command": [sys.executable, *sys.argv],
        "source_sha256": source_hashes,
        "environment": environment,
        "results": [_result_record(result) for result in results],
    }

    stem = f"{run_id}_{experiment}"
    json_path = log_dir / f"{stem}.json"
    text_path = log_dir / f"{stem}.txt"
    csv_path = log_dir / "history.csv"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    text_lines = [
        "MLP 实验日志",
        f"记录时间：{record['recorded_at']}",
        f"运行编号：{run_id}",
        f"实验：{experiment}｜模式：{mode}｜设备：{requested_device} -> {resolved_device}",
        f"MLP 源码 SHA-256：{source_hashes['mlp_lab']}",
        "",
        format_result_table(results),
    ]
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    write_header = not csv_path.exists()
    csv_mode = "w" if write_header else "a"
    csv_encoding = "utf-8-sig" if write_header else "utf-8"
    with csv_path.open(csv_mode, encoding=csv_encoding, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        for result in results:
            config = result.config
            writer.writerow(
                {
                    "recorded_at": record["recorded_at"],
                    "run_id": run_id,
                    "experiment": experiment,
                    "source_core_sha256": source_hashes["mlp_lab"],
                    "config_name": config.name,
                    "dataset": config.dataset,
                    "seed": config.seed,
                    "noise": config.noise,
                    "n_samples": config.n_samples,
                    "epochs": config.epochs,
                    "hidden_sizes": "x".join(map(str, config.hidden_sizes)),
                    "activation": config.activation,
                    "loss_name": config.loss_name,
                    "optimizer": config.optimizer,
                    "learning_rate": config.learning_rate,
                    "dropout": config.dropout,
                    "resolved_device": resolved_device,
                    "final_train_accuracy": result.train_accuracy[-1],
                    "final_test_accuracy": result.final_test_accuracy,
                    "final_train_loss": result.train_loss[-1],
                    "final_gradient_norm": result.gradient_norm[-1],
                    "parameter_count": result.parameter_count,
                    "elapsed_seconds": result.elapsed_seconds,
                }
            )

    print(f"已保存 JSON 日志：{json_path}")
    print(f"已保存终端摘要：{text_path}")
    print(f"已更新历史表：{csv_path}")
    return json_path, text_path, csv_path


# =============================================================================
# 3. 基线与组件对比执行流程
# =============================================================================


def run_baseline(args: argparse.Namespace, base: ExperimentConfig) -> None:
    """训练一个基线模型并输出两类互补可视化。

    Args:
        args: 提供 ``save_dir`` 和 ``no_show`` 输出设置。
        base: 已合并命令行参数的基线配置。

    Returns:
        ``None``。终端打印结果表；图形根据参数显示和/或保存。

    “训练故事图”强调边界随 epoch 的形成，“最终诊断图”强调误判、收敛、泛化
    和梯度。两张图回答不同问题，因此基线同时保留。
    """

    data = make_dataset(base.dataset, base.n_samples, base.noise, base.seed)
    result = train_experiment(base, data=data)
    print_result_table([result])
    if not args.no_log:
        save_experiment_log(
            [result], "baseline", args.device, str(resolve_device(args.device)), args.mode, args.log_dir
        )
    story = plot_training_story(result)
    save_or_show(story, "01_training_story.png", args.save_dir, show=not args.no_show)
    diagnosis = plot_final_diagnosis(result)
    save_or_show(diagnosis, "02_final_diagnosis.png", args.save_dir, show=not args.no_show)


def run_one_comparison(
    args: argparse.Namespace,
    base: ExperimentConfig,
    kind: str,
) -> None:
    """运行一个目标组件的全部配置并绘制横向对比矩阵。

    Args:
        args: 提供保存/显示控制。
        base: 所有非目标组件保持不变的基线配置。
        kind: activation、depth、loss、optimizer、dropout 或 width。

    Returns:
        ``None``。结果表写到终端，对比矩阵显示和/或保存。
    """

    results = run_comparison(kind, base)
    print_result_table(results)
    if not args.no_log:
        save_experiment_log(
            results, kind, args.device, str(resolve_device(args.device)), args.mode, args.log_dir
        )
    figure = plot_comparison(results, COMPARISON_TITLES[kind])
    save_or_show(figure, f"comparison_{kind}.png", args.save_dir, show=not args.no_show)


# =============================================================================
# 4. 程序入口与实验分派
# =============================================================================


def main() -> None:
    """解析命令行并分派数据画廊、基线、单项对比或全部对比。

    Returns:
        ``None``。

    在开始训练前先调用 ``resolve_device``，使显式 CUDA 错误尽早出现。``all``
    按预设顺序逐个完成六类对比，默认 ``baseline`` 不会意外触发大量训练。
    """

    args = build_parser().parse_args()
    chosen_device = resolve_device(args.device)
    print(f"运行设备：{chosen_device}｜模式：{args.mode}｜数据：{args.dataset}")

    if args.experiment == "gallery":
        figure = plot_dataset_gallery(seed=args.seed)
        save_or_show(figure, "dataset_gallery.png", args.save_dir, show=not args.no_show)
        return

    base = make_base_from_args(args)
    if args.experiment == "baseline":
        run_baseline(args, base)
        return
    if args.experiment == "all":
        for kind in COMPARISON_TITLES:
            run_one_comparison(args, base, kind)
        return
    run_one_comparison(args, base, args.experiment)


if __name__ == "__main__":
    # Windows 上必须保留入口保护；否则 DataLoader 扩展时可能重复启动进程。
    main()
