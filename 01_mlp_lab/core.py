"""Compatibility facade for existing scripts.

New notebooks should import from the mlp_lab package. The implementation is
split into small modules so learners only need to read mlp_lab/model.py when a
task asks them to change the network.
"""

import torch

from mlp_lab.api import challenge_report, compare, quick_demo, show_datasets
from mlp_lab.data import DatasetBundle, make_dataset, resolve_device, seed_everything
from mlp_lab.engine import (
    ExperimentConfig,
    TrainingResult,
    _accuracy,
    _classification_loss,
    _clone_state,
    _make_optimizer,
    _synchronize,
    base_config,
    comparison_configs,
    estimate_update_steps,
    mode_defaults,
    run_comparison,
    train_experiment,
    validate_config,
)
from mlp_lab.model import MLP, _activation
from mlp_lab.plots import (
    POINT_COLORS,
    REGION_COLORS,
    configure_chinese_font,
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
    "POINT_COLORS",
    "REGION_COLORS",
    "TrainingResult",
    "base_config",
    "challenge_report",
    "compare",
    "comparison_configs",
    "estimate_update_steps",
    "configure_chinese_font",
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
