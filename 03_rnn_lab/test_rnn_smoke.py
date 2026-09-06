"""RNN 实验室的最小可运行检查，不依赖 pytest。"""

from io import BytesIO
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rnn_lab import (
    ExperimentConfig,
    RecurrentClassifier,
    compare_cells,
    configure_chinese_font,
    input_gradient_by_time,
    make_dataloaders,
    make_delayed_recall_data,
    plot_comparison,
    plot_confusion_matrix,
    plot_hidden_states,
    plot_input_gradients,
    plot_recurrence_diagram,
    plot_sequence_sample,
    plot_tensor_structure,
    plot_training_history,
    resolve_device,
    run_experiment,
)


def main() -> None:
    assert resolve_device("cpu").type == "cpu"
    with patch("rnn_lab.data.torch.cuda.is_available", return_value=False):
        try:
            resolve_device("cuda")
        except RuntimeError as error:
            assert "检测不到 CUDA" in str(error)
        else:
            raise AssertionError("CUDA 不可用时应报错。")

    data = make_delayed_recall_data(
        sequence_length=9, train_size=60, val_size=30, test_size=30, seed=7
    )
    assert data.x_train.shape == (60, 9, 6)
    assert data.y_train.shape == (60,)
    assert torch.all(data.x_train[:, 0, :3].sum(dim=1) == 1)
    assert torch.all(data.x_train[:, 1:-1, :3] == 0)
    assert torch.all(data.x_train[:, -1, 5] == 1)
    loaders = make_dataloaders(data, batch_size=16, seed=7)
    batch_x, batch_y = next(iter(loaders["train"]))
    assert batch_x.ndim == 3 and batch_y.ndim == 1

    for cell_type in ("rnn", "lstm", "gru"):
        model = RecurrentClassifier(cell_type=cell_type, hidden_size=8)
        assert model(batch_x).shape == (len(batch_x), 3)
        assert model.sequence_features(batch_x).shape == (len(batch_x), 9, 8)

    config = ExperimentConfig(
        cell_type="lstm", sequence_length=9, hidden_size=8, epochs=2,
        train_size=60, val_size=30, test_size=30, batch_size=16, seed=7,
    )
    result = run_experiment(config, data=data)
    assert len(result["history"]["train_loss"]) == 2
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["metrics"]["predictions"].shape == (30,)
    assert result["parameter_count"] > 0
    assert torch.get_num_threads() == 1
    gradients = input_gradient_by_time(result["model"], batch_x, batch_y)
    assert gradients.shape == (9,) and torch.isfinite(gradients).all()

    comparison_base = ExperimentConfig(
        epochs=1, hidden_size=6, train_size=30, val_size=15, test_size=15,
        batch_size=15, seed=9,
    )
    comparison = compare_cells(comparison_base, lengths=(6,))
    assert [item["config"].cell_type for item in comparison] == ["rnn", "lstm", "gru"]

    configure_chinese_font()
    figures = [
        plot_sequence_sample(batch_x, batch_y),
        plot_tensor_structure(data),
        plot_recurrence_diagram(),
        plot_training_history(result["history"]),
        plot_confusion_matrix(result["metrics"]["targets"], result["metrics"]["predictions"]),
        plot_hidden_states(result["model"], batch_x),
        plot_input_gradients(result["model"], batch_x, batch_y),
        plot_comparison(comparison),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        buffer = BytesIO()
        figures[0].savefig(buffer, format="png")
    assert buffer.tell() > 0
    assert not any("Glyph" in str(item.message) for item in caught)
    plt.close("all")
    print("RNN smoke test passed.")


if __name__ == "__main__":
    main()
