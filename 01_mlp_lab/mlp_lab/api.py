"""Small, visual-first API for the notebooks."""

from dataclasses import replace
import math

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
    """交互后端显示图片；测试使用的 Agg 后端不弹出窗口。"""
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


def challenge_report(
    result: TrainingResult,
    accuracy_target: float = 0.90,
    parameter_budget: int = 500,
    update_budget: int = 400,
) -> dict[str, float | int | bool]:
    """按准确率、参数量和更新次数给一次闯关实验计星。"""
    if not 0.0 <= accuracy_target <= 1.0:
        raise ValueError("accuracy_target 必须位于 0 到 1 之间。")
    if parameter_budget <= 0 or update_budget <= 0:
        raise ValueError("parameter_budget 和 update_budget 必须大于 0。")

    update_steps = (
        math.ceil(len(result.data.x_train) / result.config.batch_size)
        * result.config.epochs
    )
    checks = {
        "accuracy_star": result.final_test_accuracy >= accuracy_target,
        "parameter_star": result.parameter_count <= parameter_budget,
        "update_star": update_steps <= update_budget,
    }
    stars = sum(checks.values())
    print("\n闯关计分")
    print(
        f"{'⭐' if checks['accuracy_star'] else '☆'} 准确率："
        f"{result.final_test_accuracy:.1%}（目标 ≥ {accuracy_target:.1%}）"
    )
    print(
        f"{'⭐' if checks['parameter_star'] else '☆'} 参数量："
        f"{result.parameter_count}（预算 ≤ {parameter_budget}）"
    )
    print(
        f"{'⭐' if checks['update_star'] else '☆'} 更新次数："
        f"{update_steps}（预算 ≤ {update_budget}）"
    )
    print(f"总计：{stars}/3 星。分数不使用运行时间，避免共享服务器负载影响公平性。")
    return {
        **checks,
        "stars": stars,
        "accuracy": result.final_test_accuracy,
        "parameters": result.parameter_count,
        "update_steps": update_steps,
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
    device: str = "cpu",
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
    device: str = "cpu",
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
