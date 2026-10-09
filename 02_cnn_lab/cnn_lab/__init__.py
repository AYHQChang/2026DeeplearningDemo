"""CNN 课堂实验室公开接口。"""

from .api import challenge_report, compare_pooling, quick_demo, show_convolution, show_data
from .data import ImageDatasetBundle, load_digits_data, resolve_device, seed_everything
from .engine import (
    ExperimentConfig, TrainingResult, base_config, compare_pooling_experiments,
    estimate_update_steps, pooling_configs, shift_images_right, shifted_accuracy,
    train_experiment, validate_config, evaluate_final, evaluation_data,
    experiment_records, save_experiment_records,
)
from .model import ComposableCNN, ConvBlock, SmallCNN, print_model_summary, trace_forward
from .operators import show_convolution_step, show_pooling_step
from .plots import (
    configure_chinese_font, format_result_table, plot_convolution_demo,
    plot_digit_gallery, plot_feature_maps, plot_forward_pass, plot_pixel_and_shape,
    plot_pooling_comparison, plot_pooling_demo, plot_training_overview,
    print_result_table,
)

__all__ = [
    "ExperimentConfig", "ImageDatasetBundle", "SmallCNN", "TrainingResult",
    "ComposableCNN", "ConvBlock", "print_model_summary", "trace_forward", "evaluate_final",
    "evaluation_data", "experiment_records", "save_experiment_records",
    "base_config", "challenge_report", "compare_pooling", "compare_pooling_experiments",
    "configure_chinese_font", "format_result_table", "load_digits_data",
    "plot_convolution_demo", "plot_digit_gallery", "plot_feature_maps", "plot_forward_pass",
    "plot_pixel_and_shape", "plot_pooling_comparison", "plot_pooling_demo",
    "estimate_update_steps", "plot_training_overview", "pooling_configs", "print_result_table",
    "quick_demo", "resolve_device", "seed_everything", "shift_images_right",
    "shifted_accuracy", "show_convolution", "show_data", "train_experiment",
    "show_convolution_step", "show_pooling_step",
    "validate_config",
]
