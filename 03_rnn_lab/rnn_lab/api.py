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
    """运行一次可直接观察的延迟回忆实验。"""

    config = ExperimentConfig(
        name=f"{cell_type.upper()}｜长度 {sequence_length}",
        cell_type=cell_type,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    result = run_experiment(config)
    print_result_table([result])
    test_x, test_y = next(iter(result["loaders"]["test"]))
    plot_sequence_sample(test_x, test_y)
    plot_training_history(result["history"], title=config.name)
    plot_confusion_matrix(result["metrics"]["targets"], result["metrics"]["predictions"])
    _show()
    return result


def compare(
    lengths: tuple[int, ...] = (12, 30),
    epochs: int = 8,
    seed: int = 42,
    device: str = "cpu",
) -> list[dict[str, object]]:
    """以相同超参数比较三种循环单元和给定序列长度。"""

    base = replace(
        ExperimentConfig(seed=seed, device=device),
        epochs=epochs,
        train_size=600,
        val_size=180,
        test_size=180,
    )
    results = compare_cells(base, lengths=lengths)
    print_result_table(results)
    plot_comparison(results)
    _show()
    return results
