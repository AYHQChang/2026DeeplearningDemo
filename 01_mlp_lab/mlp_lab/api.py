"""Small, visual-first API for the notebooks."""

from dataclasses import replace

import matplotlib.pyplot as plt

from .data import make_dataset
from .engine import TrainingResult, base_config, run_comparison, train_experiment
from .plots import (
    plot_comparison,
    plot_dataset_gallery,
    plot_final_diagnosis,
    plot_training_story,
    print_result_table,
)


def _show() -> None:
    if plt.get_backend().lower() != "agg":
        plt.show()


COMPARISON_TITLES = {
    "activation": "激活函数对比：非线性形状如何影响学习",
    "depth": "网络深度对比：更深是否一定更好",
    "loss": "损失函数对比：分类目标应该怎样表达",
    "optimizer": "优化器对比：参数怎样更新",
    "dropout": "Dropout 对比：正则化与欠拟合",
    "width": "隐藏层宽度对比：容量与泛化",
}


def show_datasets(seed: int = 42, n_samples: int = 450):
    """Display the three built-in two-dimensional datasets."""
    figure = plot_dataset_gallery(seed=seed, n_samples=n_samples)
    _show()
    return figure


def quick_demo(
    dataset: str = "moons",
    hidden_sizes: tuple[int, ...] = (16, 16),
    activation: str = "relu",
    optimizer: str = "adam",
    learning_rate: float = 0.02,
    dropout: float = 0.0,
    epochs: int = 40,
    n_samples: int = 360,
    seed: int = 42,
    device: str = "auto",
    show_story: bool = True,
) -> TrainingResult:
    """Train one model, print its summary and immediately display the result."""
    config = replace(
        base_config(dataset=dataset, mode="fast", seed=seed, device=device),
        name=f"{dataset}｜{activation}｜{optimizer}",
        hidden_sizes=tuple(hidden_sizes),
        activation=activation,
        optimizer=optimizer,
        learning_rate=learning_rate,
        dropout=dropout,
        epochs=epochs,
        n_samples=n_samples,
    )
    data = make_dataset(config.dataset, config.n_samples, config.noise, config.seed)
    result = train_experiment(config, data=data)
    print_result_table([result])
    if show_story:
        plot_training_story(result)
    plot_final_diagnosis(result)
    _show()
    return result


def compare(
    component: str = "activation",
    dataset: str = "moons",
    epochs: int = 35,
    n_samples: int = 360,
    seed: int = 42,
    device: str = "auto",
) -> list[TrainingResult]:
    """Run one controlled component comparison and display its figure."""
    if component not in COMPARISON_TITLES:
        choices = "、".join(COMPARISON_TITLES)
        raise ValueError(f"component 只能是：{choices}。")
    base = replace(
        base_config(dataset=dataset, mode="fast", seed=seed, device=device),
        epochs=epochs,
        n_samples=n_samples,
    )
    results = run_comparison(component, base)
    print_result_table(results)
    plot_comparison(results, COMPARISON_TITLES[component])
    _show()
    return results
