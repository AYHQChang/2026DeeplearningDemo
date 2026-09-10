"""小型 Transformer 注意力寻宝课堂实验室公开接口。"""

from .api import challenge_report, compare, quick_demo
from .data import (
    CLASS_NAMES, QUERY_ID, TOKEN_NAMES, RetrievalDatasetBundle,
    make_dataloaders, make_retrieval_data, resolve_device, seed_everything,
)
from .engine import (
    ExperimentConfig, attention_budget, attention_for_sample, compare_experiments,
    comparison_configs, count_parameters, evaluate_length_transfer, evaluate_model,
    predict_proba, run_experiment, train_model, update_count, validate_config,
)
from .model import (
    SinusoidalPositionalEncoding, TinyTransformerClassifier,
    estimate_parameter_count, scaled_dot_product_attention,
)
from .plots import (
    configure_chinese_font, plot_attention_weights, plot_comparison,
    plot_confusion_matrix, plot_model_attention, plot_positional_encoding,
    plot_tensor_structure, plot_token_sequence, plot_training_history,
    print_result_table,
)

__all__ = [
    "CLASS_NAMES", "QUERY_ID", "TOKEN_NAMES", "ExperimentConfig",
    "RetrievalDatasetBundle", "SinusoidalPositionalEncoding",
    "TinyTransformerClassifier", "attention_budget", "attention_for_sample",
    "challenge_report", "compare", "compare_experiments", "comparison_configs",
    "configure_chinese_font", "count_parameters", "estimate_parameter_count",
    "evaluate_length_transfer", "evaluate_model", "make_dataloaders",
    "make_retrieval_data", "plot_attention_weights", "plot_comparison",
    "plot_confusion_matrix", "plot_model_attention", "plot_positional_encoding",
    "plot_tensor_structure", "plot_token_sequence", "plot_training_history",
    "predict_proba", "print_result_table", "quick_demo", "resolve_device",
    "run_experiment", "scaled_dot_product_attention", "seed_everything",
    "train_model", "update_count", "validate_config",
]
