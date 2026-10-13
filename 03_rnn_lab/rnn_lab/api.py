"""Notebook 和命令行使用的短入口。"""

from dataclasses import replace

import matplotlib.pyplot as plt

from .engine import ExperimentConfig, compare_cells, run_experiment
from .plots import (
    plot_comparison,
    plot_confusion_matrix,
    plot_sequence_sample,
    plot_training_history,
    print_result_table,
)


def _show() -> None:
    if plt.get_backend().lower() != "agg":
        plt.show()


def quick_demo(
    cell_type: str = "lstm",
    sequence_length: int = 30,
    hidden_size: int = 24,
    epochs: int = 12,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, object]:
    """运行一次延迟回忆实验，只用验证集进行教学观察。"""

    config = ExperimentConfig(
        name=f"{cell_type.upper()}｜长度 {sequence_length}",
        cell_type=cell_type,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    result = run_experiment(config, evaluate_test=False)
    print_result_table([result])
    val_x, val_y = next(iter(result["loaders"]["val"]))
    plot_sequence_sample(val_x, val_y, class_names=result["data"].class_names)
    plot_training_history(result["history"], title=config.name, n_classes=result["data"].n_classes)
    plot_confusion_matrix(
        result["val_metrics"]["targets"], result["val_metrics"]["predictions"],
        class_names=result["data"].class_names, title="验证集混淆矩阵",
    )
    _show()
    return result


def compare(
    lengths: tuple[int, ...] = (12, 30),
    epochs: int = 8,
    seed: int = 42,
    device: str = "cpu",
) -> list[dict[str, object]]:
    """以相同超参数比较三种循环单元，仅查看验证集结果。"""

    base = replace(
        ExperimentConfig(seed=seed, device=device),
        epochs=epochs,
        train_size=600,
        val_size=180,
        test_size=180,
    )
    results = compare_cells(base, lengths=lengths, evaluate_test=False)
    print_result_table(results)
    plot_comparison(results, split="val")
    _show()
    return results
