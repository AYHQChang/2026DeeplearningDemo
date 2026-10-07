"""Notebook 使用的短入口：数据、卷积机制、训练与池化对比。"""

from dataclasses import replace
import math

import matplotlib.pyplot as plt
from torch import nn

from .data import ImageDatasetBundle, load_digits_data
from .engine import (
    TrainingResult,
    base_config,
    compare_pooling_experiments,
    evaluation_data,
    shifted_accuracy,
    train_experiment,
)
from .model import ConvBlock
from .plots import (
    plot_convolution_demo,
    plot_digit_gallery,
    plot_feature_maps,
    plot_pixel_and_shape,
    plot_pooling_comparison,
    plot_pooling_demo,
    plot_training_overview,
    print_result_table,
)


def _show() -> None:
    if plt.get_backend().lower() != "agg":
        plt.show()


def challenge_report(
    result: TrainingResult,
    accuracy_target: float = 0.96,
    shifted_target: float = 0.60,
    parameter_budget: int = 4_000,
    update_budget: int = 300,
) -> dict[str, float | int | bool]:
    """按识别、平移鲁棒性、参数量和更新次数给 CNN 实验计徽章。"""
    if not 0.0 <= accuracy_target <= 1.0 or not 0.0 <= shifted_target <= 1.0:
        raise ValueError("accuracy_target 和 shifted_target 必须位于 0 到 1 之间。")
    if parameter_budget <= 0 or update_budget <= 0:
        raise ValueError("parameter_budget 和 update_budget 必须大于 0。")

    _, _, evaluation_label = evaluation_data(result)
    accuracy = (
        result.final_test_accuracy if result.test_accuracy
        else result.final_validation_accuracy
    )
    shifted = shifted_accuracy(result)
    update_steps = (
        math.ceil(len(result.data.x_train) / result.config.batch_size)
        * result.config.epochs
    )
    checks = {
        "accuracy_badge": accuracy >= accuracy_target,
        "shift_badge": shifted >= shifted_target,
        "parameter_badge": result.parameter_count <= parameter_budget,
        "update_badge": update_steps <= update_budget,
    }
    badges = sum(checks.values())
    print("\nCNN 闯关计分")
    print(
        f"{'🏅' if checks['accuracy_badge'] else '○'} 识别："
        f"{accuracy:.1%}（{evaluation_label}集，目标 ≥ {accuracy_target:.1%}）"
    )
    print(
        f"{'🏅' if checks['shift_badge'] else '○'} 平移："
        f"{shifted:.1%}（目标 ≥ {shifted_target:.1%}）"
    )
    print(
        f"{'🏅' if checks['parameter_badge'] else '○'} 轻量："
        f"{result.parameter_count:,}（预算 ≤ {parameter_budget:,}）"
    )
    print(
        f"{'🏅' if checks['update_badge'] else '○'} 节能："
        f"{update_steps} 次更新（预算 ≤ {update_budget}）"
    )
    print(f"总计：{badges}/4 枚徽章。运行时间不计分，避免服务器负载影响公平性。")
    return {
        **checks,
        "badges": badges,
        "accuracy": accuracy,
        "shifted_accuracy": shifted,
        "parameters": result.parameter_count,
        "update_steps": update_steps,
    }


def show_data(data: ImageDatasetBundle | None = None) -> tuple[plt.Figure, plt.Figure]:
    data = data or load_digits_data()
    figures = (plot_digit_gallery(data), plot_pixel_and_shape(data))
    _show()
    return figures


def show_convolution(data: ImageDatasetBundle | None = None) -> tuple[plt.Figure, plt.Figure]:
    data = data or load_digits_data()
    figures = (plot_convolution_demo(data), plot_pooling_demo(data))
    _show()
    return figures


def quick_demo(
    channels: tuple[int, int] = (8, 16),
    kernel_size: int = 3,
    pooling: str = "max",
    activation: str = "relu",
    dropout: float = 0.0,
    learning_rate: float = 0.01,
    epochs: int = 20,
    seed: int = 42,
    device: str = "cpu",
    show_features: bool = True,
    data: ImageDatasetBundle | None = None,
    blocks: tuple[ConvBlock, ...] | None = None,
    model: nn.Module | None = None,
    evaluate_test: bool = True,
    batch_size: int = 64,
) -> TrainingResult:
    """共用数字与自定义图片数据；blocks 或 model 可替换网络结构。"""

    if model is not None:
        name = f"自定义 CNN｜{type(model).__name__}"
    elif blocks is not None:
        name = f"组合 CNN｜{len(blocks)} 个卷积块"
    else:
        name = f"CNN｜{activation}｜通道 {'→'.join(map(str, channels))}｜{pooling} pooling"
    config = replace(
        base_config(seed=seed, device=device),
        name=name,
        channels=channels,
        kernel_size=kernel_size,
        pooling=pooling,
        activation=activation,
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        blocks=blocks,
    )
    result = train_experiment(config, data=data, model=model, evaluate_test=evaluate_test)
    print_result_table([result])
    plot_training_overview(result)
    if show_features:
        if callable(getattr(result.model, "feature_maps", None)):
            plot_feature_maps(result)
        else:
            print("这个自定义模型没有 feature_maps()，已跳过特征图。训练曲线、预测与混淆矩阵仍可查看。")
    _show()
    return result


def compare_pooling(
    channels: tuple[int, int] = (8, 16),
    epochs: int = 12,
    seed: int = 42,
    device: str = "cpu",
    data: ImageDatasetBundle | None = None,
    blocks: tuple[ConvBlock, ...] | None = None,
    evaluate_test: bool = True,
    batch_size: int = 64,
) -> list[TrainingResult]:
    """只更换第一个卷积块后的池化，其余结构与数据保持一致。"""

    base = replace(
        base_config(seed=seed, device=device),
        channels=channels, epochs=epochs, blocks=blocks, batch_size=batch_size,
    )
    results = compare_pooling_experiments(base, data=data, evaluate_test=evaluate_test)
    print_result_table(results)
    plot_pooling_comparison(results)
    _show()
    return results
