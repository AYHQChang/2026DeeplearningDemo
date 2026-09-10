"""Notebook 和命令行使用的短入口。"""

from dataclasses import replace

import matplotlib.pyplot as plt

from .engine import (
    ExperimentConfig,
    compare_experiments,
    evaluate_length_transfer,
    run_experiment,
)
from .plots import (
    plot_comparison,
    plot_confusion_matrix,
    plot_model_attention,
    plot_token_sequence,
    plot_training_history,
    print_result_table,
)


def _show() -> None:
    if plt.get_backend().lower() != "agg":
        plt.show()


def quick_demo(
    content_length: int = 12,
    d_model: int = 32,
    num_heads: int = 4,
    num_layers: int = 1,
    dim_feedforward: int = 64,
    dropout: float = 0.1,
    activation: str = "gelu",
    use_positional_encoding: bool = True,
    learning_rate: float = 0.003,
    epochs: int = 12,
    seed: int = 42,
    device: str = "cpu",
    show_attention: bool = True,
) -> dict[str, object]:
    config = ExperimentConfig(
        name="有位置编码" if use_positional_encoding else "无位置编码",
        content_length=content_length, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, dim_feedforward=dim_feedforward,
        dropout=dropout, activation=activation,
        use_positional_encoding=use_positional_encoding,
        learning_rate=learning_rate, epochs=epochs, seed=seed, device=device,
    )
    result = run_experiment(config)
    print_result_table([result])
    test_x, test_y = next(iter(result["loaders"]["test"]))
    plot_token_sequence(test_x, test_y)
    plot_training_history(result["history"], title=config.name)
    plot_confusion_matrix(result["metrics"]["targets"], result["metrics"]["predictions"])
    if show_attention:
        plot_model_attention(result["model"], test_x, device=device)
    _show()
    return result


def compare(
    epochs: int = 12,
    seed: int = 42,
    device: str = "cpu",
) -> list[dict[str, object]]:
    base = replace(ExperimentConfig(seed=seed, device=device), epochs=epochs)
    results = compare_experiments(base)
    print_result_table(results)
    plot_comparison(results)
    _show()
    return results


def challenge_report(
    result: dict[str, object],
    accuracy_target: float = 0.85,
    transfer_target: float = 0.45,
    parameter_budget: int = 15_000,
    attention_budget_target: int = 3_000_000,
    transfer_content_length: int = 24,
    transfer_test_size: int = 300,
) -> dict[str, object]:
    """按准确率、长度迁移、参数量和组合预算发放四枚徽章。"""

    transfer = evaluate_length_transfer(
        result, content_length=transfer_content_length, test_size=transfer_test_size
    )
    badges = {
        "寻宝": result["metrics"]["accuracy"] >= accuracy_target,
        "远行": transfer["accuracy"] >= transfer_target,
        "轻装": result["parameter_count"] <= parameter_budget,
        "节能": result["attention_budget"] <= attention_budget_target,
    }
    print("\n四徽章报告（不使用运行时间计分）")
    print(f"测试 Accuracy：{result['metrics']['accuracy']:.1%}｜目标 {accuracy_target:.1%}")
    print(f"长度 {transfer_content_length} 迁移：{transfer['accuracy']:.1%}｜目标 {transfer_target:.1%}")
    print(f"参数量：{result['parameter_count']:,}｜预算 {parameter_budget:,}")
    print(f"Attention 组合预算：{result['attention_budget']:,}｜预算 {attention_budget_target:,}")
    print("徽章：", " ".join(f"{'✅' if won else '⬜'}{name}" for name, won in badges.items()))
    return {
        "badges": badges,
        "badge_count": sum(badges.values()),
        "transfer_metrics": transfer,
    }
