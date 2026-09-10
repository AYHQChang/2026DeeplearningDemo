"""Transformer 注意力寻宝实验室的最小可运行检查。"""

from dataclasses import replace
from io import BytesIO
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from transformer_lab import (
    ExperimentConfig,
    SinusoidalPositionalEncoding,
    TinyTransformerClassifier,
    attention_for_sample,
    compare_experiments,
    configure_chinese_font,
    estimate_parameter_count,
    evaluate_length_transfer,
    make_dataloaders,
    make_retrieval_data,
    plot_attention_weights,
    plot_comparison,
    plot_confusion_matrix,
    plot_model_attention,
    plot_positional_encoding,
    plot_tensor_structure,
    plot_token_sequence,
    plot_training_history,
    resolve_device,
    run_experiment,
    scaled_dot_product_attention,
    validate_config,
)


def _rows(tensor: torch.Tensor) -> set[tuple[int, ...]]:
    return {tuple(row.tolist()) for row in tensor}


def main() -> None:
    assert resolve_device("cpu").type == "cpu"
    for invalid in ("auto", "cuda"):
        try:
            resolve_device(invalid)
        except ValueError as error:
            assert "不允许 auto" in str(error)
        else:
            raise AssertionError("共享入口不应接受自动或无编号 GPU。")
    with patch("transformer_lab.data.torch.cuda.is_available", return_value=False):
        try:
            resolve_device("cuda:0")
        except RuntimeError as error:
            assert "检测不到 CUDA" in str(error)
        else:
            raise AssertionError("CUDA 不可用时应报错。")

    data = make_retrieval_data(
        content_length=9, train_size=60, val_size=30, test_size=30, seed=7
    )
    assert data.x_train.shape == (60, 10)
    assert torch.equal(data.x_train[:, 0], data.y_train)
    assert torch.all(data.x_train[:, -1] == 3)
    counts = torch.stack([(data.x_train[:, :-1] == token).sum(dim=1) for token in range(3)])
    assert torch.all(counts == 3)
    train_rows, val_rows, test_rows = map(_rows, (data.x_train, data.x_val, data.x_test))
    assert train_rows.isdisjoint(val_rows | test_rows) and val_rows.isdisjoint(test_rows)
    loaders = make_dataloaders(data, batch_size=16, seed=7)
    assert loaders["train"].num_workers == 0
    batch_x, batch_y = next(iter(loaders["train"]))

    model = TinyTransformerClassifier(
        d_model=8, num_heads=2, num_layers=1, dim_feedforward=16,
        dropout=0.0, max_length=25,
    )
    logits, attention = model.forward_with_attention(batch_x)
    assert logits.shape == (len(batch_x), 3)
    assert attention.shape == (len(batch_x), 2, 10, 10)
    assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=1e-5)
    no_position = TinyTransformerClassifier(
        d_model=8, num_heads=2, num_layers=1, dim_feedforward=16,
        dropout=0.0, use_positional_encoding=False,
    ).eval()
    permuted = batch_x[:1].clone()
    permuted[:, :-1] = permuted[:, :-1].flip(dims=(1,))
    assert torch.allclose(no_position(batch_x[:1]), no_position(permuted), atol=1e-6)
    try:
        TinyTransformerClassifier(d_model=8, num_heads=0)
    except ValueError as error:
        assert "必须大于 0" in str(error)
    else:
        raise AssertionError("无效模型维度应该被拒绝。")
    position = SinusoidalPositionalEncoding(8, max_length=25)
    assert position(torch.zeros(2, 10, 8)).shape == (2, 10, 8)

    query = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    scores, weights, output = scaled_dot_product_attention(query, keys, values)
    assert scores.shape == weights.shape == (2,) and output.shape == (2,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))

    config = ExperimentConfig(
        content_length=9, d_model=8, num_heads=2, dim_feedforward=16,
        dropout=0.0, epochs=2, train_size=60, val_size=30,
        test_size=30, batch_size=16, seed=7,
    )
    result = run_experiment(config, data=data)
    assert len(result["history"]["train_loss"]) == 2
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["parameter_count"] == estimate_parameter_count(8, 1, 16)
    assert torch.get_num_threads() == 1
    try:
        run_experiment(replace(config, train_size=61), data=data)
    except ValueError as error:
        assert "传入数据量" in str(error)
    else:
        raise AssertionError("自定义数据不能绕过配置中的数据量边界。")
    probabilities, selected_attention = attention_for_sample(result["model"], batch_x)
    assert probabilities.shape == (1, 3) and selected_attention.shape == (1, 2, 10, 10)
    transfer = evaluate_length_transfer(result, content_length=12, test_size=30)
    assert 0.0 <= transfer["accuracy"] <= 1.0

    comparison_base = replace(
        config, epochs=1, train_size=30, val_size=15, test_size=15, batch_size=15
    )
    comparison = compare_experiments(comparison_base)
    assert len(comparison) == 4

    unsafe_cases = (
        (replace(config, device="auto"), "不允许 auto"),
        (replace(config, content_length=27), "不能超过 24"),
        (replace(config, train_size=3001), "train_size"),
        (replace(config, epochs=31), "epochs"),
        (replace(config, d_model=30, num_heads=4), "整除"),
        (replace(config, num_layers=3), "num_layers"),
        (replace(config, dim_feedforward=129), "dim_feedforward"),
        (replace(config, train_size=3000, batch_size=1, epochs=1), "更新次数"),
        (replace(config, content_length=24, train_size=1200, epochs=12, num_layers=2), "组合预算"),
    )
    for unsafe, message in unsafe_cases:
        try:
            validate_config(unsafe)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"超出预算的配置应该被拒绝：{unsafe}")

    configure_chinese_font()
    figures = [
        plot_token_sequence(batch_x, batch_y),
        plot_tensor_structure(data),
        plot_positional_encoding(d_model=8, max_length=12),
        plot_attention_weights(weights, labels=["位置0", "位置1"]),
        plot_training_history(result["history"]),
        plot_confusion_matrix(result["metrics"]["targets"], result["metrics"]["predictions"]),
        plot_model_attention(result["model"], batch_x),
        plot_comparison(comparison),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        buffer = BytesIO()
        figures[0].savefig(buffer, format="png")
    assert buffer.tell() > 0
    assert not any("Glyph" in str(item.message) for item in caught)
    plt.close("all")
    print("Transformer smoke test passed.")


if __name__ == "__main__":
    main()
