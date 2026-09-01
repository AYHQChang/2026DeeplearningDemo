"""Notebook 使用的短入口：数据、卷积机制、训练与池化对比。"""

from dataclasses import replace

import matplotlib.pyplot as plt

from .data import ImageDatasetBundle, load_digits_data
from .engine import TrainingResult, base_config, compare_pooling_experiments, train_experiment
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
    dropout: float = 0.0,
    learning_rate: float = 0.01,
    epochs: int = 20,
    seed: int = 42,
    device: str = "cpu",
    show_features: bool = True,
) -> TrainingResult:
    config = replace(
        base_config(seed=seed, device=device),
        name=f"CNN｜通道 {channels[0]}→{channels[1]}｜{pooling} pooling",
        channels=channels,
        kernel_size=kernel_size,
        pooling=pooling,
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=epochs,
    )
    result = train_experiment(config)
    print_result_table([result])
    plot_training_overview(result)
    if show_features:
        plot_feature_maps(result)
    _show()
    return result


def compare_pooling(
    channels: tuple[int, int] = (8, 16),
    epochs: int = 12,
    seed: int = 42,
    device: str = "cpu",
) -> list[TrainingResult]:
    base = replace(base_config(seed=seed, device=device), channels=channels, epochs=epochs)
    results = compare_pooling_experiments(base)
    print_result_table(results)
    plot_pooling_comparison(results)
    _show()
    return results
