"""Public API for the MLP teaching lab."""

from .api import compare, quick_demo, show_datasets
from .data import DatasetBundle, make_dataset, resolve_device, seed_everything
from .engine import (
    ExperimentConfig,
    TrainingResult,
    base_config,
    comparison_configs,
    mode_defaults,
    run_comparison,
    train_experiment,
    validate_config,
)
from .model import MLP
from .plots import (
    format_result_table,
    plot_comparison,
    plot_dataset_gallery,
    plot_final_diagnosis,
    plot_training_story,
    print_result_table,
    save_or_show,
)

__all__ = [
    "DatasetBundle",
    "ExperimentConfig",
    "MLP",
    "TrainingResult",
    "base_config",
    "compare",
    "comparison_configs",
    "format_result_table",
    "make_dataset",
    "mode_defaults",
    "plot_comparison",
    "plot_dataset_gallery",
    "plot_final_diagnosis",
    "plot_training_story",
    "print_result_table",
    "quick_demo",
    "resolve_device",
    "run_comparison",
    "save_or_show",
    "seed_everything",
    "show_datasets",
    "train_experiment",
    "validate_config",
]
