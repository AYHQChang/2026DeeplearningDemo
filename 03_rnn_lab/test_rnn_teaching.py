"""RNN 教学功能的 CPU 回归检查；只用随机序列，不包含图片作业答案。"""

from contextlib import redirect_stdout
import csv
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from rnn_lab import (
    ExperimentConfig, RecurrentClassifier, SequenceDatasetBundle,
    compare, configure_chinese_font, evaluate_final, experiment_records,
    export_experiments_csv, input_gradient_by_time, make_dataloaders,
    observe_recurrence, plot_recurrent_trace, print_model_summary,
    quick_demo, run_experiment,
)


def snapshot(model):
    return (
        {name: module.training for name, module in model.named_modules()},
        {name: value.detach().clone() for name, value in model.state_dict().items()},
        {name: None if p.grad is None else p.grad.detach().clone()
         for name, p in model.named_parameters()},
    )


def assert_unchanged(model, before):
    modes, state, gradients = before
    assert modes == {name: module.training for name, module in model.named_modules()}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        if gradients[name] is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)


def seed_existing_gradients(model):
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, 0.125) if index % 2 else None


def test_observation():
    checked = 0
    for cell in ("rnn", "lstm", "gru"):
        for layers in (1, 2):
            for steps, features, classes in ((5, 3, 2), (11, 7, 5)):
                model = RecurrentClassifier(
                    input_size=features, hidden_size=6, n_classes=classes,
                    cell_type=cell, num_layers=layers, dropout=0.3,
                )
                model.train()
                model.classifier.eval()  # 有意保留混合模式，不能仅恢复顶层 model.training。
                seed_existing_gradients(model)
                x = torch.randn(3, steps, features)
                x_before = x.clone()
                before = snapshot(model)
                trace = observe_recurrence(model, x, index=1)
                assert_unchanged(model, before)
                torch.testing.assert_close(x, x_before, rtol=0, atol=0)
                with redirect_stdout(StringIO()):
                    rows = print_model_summary(model, x)
                assert_unchanged(model, before)
                assert rows == trace["rows"]

                model.eval()
                with torch.no_grad():
                    outputs, state = model.recurrent(x)
                    logits = model(x)
                h_n = state[0] if cell == "lstm" else state
                torch.testing.assert_close(trace["hidden_states"], outputs[1], rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(trace["h_n"], h_n, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(trace["logits"], logits)
                torch.testing.assert_close(trace["probabilities"], logits.softmax(dim=1))
                torch.testing.assert_close(trace["probabilities"].sum(dim=1), torch.ones(3))
                assert torch.equal(trace["predictions"], logits.argmax(dim=1))
                assert trace["input_shape"] == (3, steps, features)
                assert trace["recurrent_output_shape"] == (3, steps, 6)
                assert trace["output_shape"] == (3, classes)
                assert trace["h_n_shape"] == (layers, 3, 6)
                assert trace["all_hidden_states"].shape == (steps, layers, 6)
                torch.testing.assert_close(trace["all_hidden_states"][-1], h_n[:, 1], rtol=1e-4, atol=1e-5)
                if cell == "lstm":
                    assert trace["c_n_shape"] == (layers, 3, 6)
                    assert trace["all_cell_states"].shape == (steps, layers, 6)
                    torch.testing.assert_close(trace["c_n"], state[1])
                    torch.testing.assert_close(trace["all_cell_states"][-1], state[1][:, 1], rtol=1e-4, atol=1e-5)
                else:
                    assert trace["c_n"] is None and trace["cell_states"] is None
                checked += 1
    print(f"PASS: {checked} recurrent shape/state combinations agree with real forward propagation.")


def test_observation_side_effects_and_plot():
    model = RecurrentClassifier(input_size=4, hidden_size=8, n_classes=5, cell_type="lstm", num_layers=2)
    model.classifier = nn.Sequential(nn.BatchNorm1d(8), nn.Dropout(0.4), nn.Linear(8, 5))
    model.train()
    model.recurrent.eval()
    model.classifier[1].eval()
    seed_existing_gradients(model)
    before = snapshot(model)
    x = torch.randn(3, 7, 4)
    trace = observe_recurrence(model, x, index=2)
    assert_unchanged(model, before)
    gradients = input_gradient_by_time(model, x, torch.tensor([0, 1, 2]))
    assert gradients.shape == (7,) and torch.isfinite(gradients).all()
    assert_unchanged(model, before)
    configure_chinese_font()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure = plot_recurrent_trace(trace, max_units=5)
        stream = BytesIO()
        figure.savefig(stream, format="png")
        plt.close(figure)
    assert stream.tell() > 0
    assert not any("Glyph" in str(item.message) for item in caught)
    assert_unchanged(model, before)
    for invalid_x, invalid_index in ((torch.randn(2, 7), 0), (torch.randn(2, 7, 9), 0), (x, 3)):
        try:
            observe_recurrence(model, invalid_x, index=invalid_index)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid sequence shape/index must give a readable ValueError.")
        assert_unchanged(model, before)
    print("PASS: observers preserve mixed modes, BatchNorm buffers and existing gradients; plot renders without missing glyphs.")


class GuardedTestLoader:
    """阻止探索阶段读取测试样本，并统计最终测试的实际迭代次数。"""

    def __init__(self, loader):
        self.loader = loader
        self.allowed = False
        self.iterations = 0

    def __iter__(self):
        if not self.allowed:
            raise AssertionError("Exploration unexpectedly read the test split.")
        self.iterations += 1
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def random_bundle(steps=7, features=5, classes=4):
    generator = torch.Generator().manual_seed(17)
    splits = []
    for size in (24, 12, 12):
        splits.extend((torch.randn(size, steps, features, generator=generator), torch.arange(size) % classes))
    return SequenceDatasetBundle(*splits, class_names=tuple(f"类别 {i}" for i in range(classes)))


def test_experiment_boundary_and_csv():
    data = random_bundle()
    config = ExperimentConfig(
        name="随机序列验证", sequence_length=7, hidden_size=6, epochs=1,
        train_size=24, val_size=12, test_size=12, batch_size=6, device="cpu", seed=17,
    )
    loaders = make_dataloaders(data, batch_size=6, seed=17)
    guard = GuardedTestLoader(loaders["test"])
    loaders["test"] = guard
    # 与图片作业无关的外部模型，用于确认训练入口真的接受调用者传入的对象。
    custom_model = nn.Sequential(nn.Flatten(), nn.Linear(7 * 5, 4))
    initial_parameters = [parameter.detach().clone() for parameter in custom_model.parameters()]
    with patch("rnn_lab.engine.make_dataloaders", return_value=loaders):
        result = run_experiment(config, data=data, model=custom_model, evaluate_test=False)
    assert result["model"] is custom_model
    assert any(not torch.equal(old, new) for old, new in zip(initial_parameters, custom_model.parameters()))
    assert result["metrics"] is None and guard.iterations == 0
    assert result["val_metrics"]["predictions"].shape == (12,)
    assert result["val_metrics"]["probabilities"].shape == (12, 4)
    result["representation"] = "随机序列"
    result["readout"] = "外部模型"
    records_before = experiment_records([result])
    assert len(records_before) == 1
    assert records_before[0]["test_accuracy"] is None
    assert records_before[0]["test_evaluated"] is False
    assert records_before[0]["representation"] == "随机序列"
    assert records_before[0]["readout"] == "外部模型"
    assert records_before[0]["test_loss"] is None
    assert records_before[0]["val_accuracy"] == result["val_metrics"]["accuracy"]
    assert records_before[0]["input_size"] == 5 and records_before[0]["n_classes"] == 4
    with TemporaryDirectory() as directory:
        exploration_path = Path(directory) / "探索阶段.csv"
        export_experiments_csv([result], exploration_path)
        with exploration_path.open(encoding="utf-8-sig", newline="") as stream:
            exploration_row = next(csv.DictReader(stream))
        assert exploration_row["test_accuracy"] == exploration_row["test_loss"] == ""
        assert exploration_row["test_evaluated"] == "False"
        assert float(exploration_row["val_accuracy"]) == result["val_metrics"]["accuracy"]
        assert guard.iterations == 0

    guard.allowed = True
    final_metrics = evaluate_final(result)
    assert result["metrics"] is final_metrics
    assert final_metrics["predictions"].shape == (12,)
    assert 0 <= final_metrics["accuracy"] <= 1
    assert guard.iterations == 1
    with patch("rnn_lab.engine.evaluate_model", side_effect=AssertionError("Final test should be cached.")):
        assert evaluate_final(result) is final_metrics
    assert guard.iterations == 1
    records_after = experiment_records([result])
    assert records_before != records_after
    assert records_after[0]["test_accuracy"] == final_metrics["accuracy"]
    assert records_after[0]["test_evaluated"] is True
    assert records_after[0]["val_accuracy"] == records_before[0]["val_accuracy"]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "实验记录.csv"
        export_experiments_csv([result], path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert set(rows[0]) == set(records_after[0])
        assert "随机序列验证" in rows[0].values()
        assert float(rows[0]["test_accuracy"]) == final_metrics["accuracy"]
        assert rows[0]["test_evaluated"] == "True"
        assert guard.iterations == 1
    print("PASS: exploration never reads test samples; final test is cached; custom model trains and CSV exports.")


def test_public_teaching_entries():
    """便捷入口的训练、画图和打印都不应读取测试集。"""

    data = random_bundle(steps=6, features=6, classes=3)
    guards = []

    def guarded_loaders(*args, **kwargs):
        loaders = make_dataloaders(*args, **kwargs)
        guard = GuardedTestLoader(loaders["test"])
        guards.append(guard)
        loaders["test"] = guard
        return loaders

    with (
        patch("rnn_lab.engine.make_delayed_recall_data", return_value=data),
        patch("rnn_lab.engine.make_dataloaders", side_effect=guarded_loaders),
        patch("rnn_lab.engine.evaluate_final", side_effect=AssertionError("Teaching entries must not evaluate test data.")),
        redirect_stdout(StringIO()),
    ):
        demo_result = quick_demo(cell_type="rnn", sequence_length=6, hidden_size=6, epochs=1)
        comparison = compare(lengths=(6,), epochs=1)
    results = [demo_result, *comparison]
    assert len(results) == len(guards) == 4
    assert all(result["metrics"] is None for result in results)
    assert all(result["val_metrics"]["targets"].shape == (12,) for result in results)
    assert all(guard.iterations == 0 for guard in guards)
    plt.close("all")
    print("PASS: quick_demo and compare train and visualize without evaluating or iterating test data.")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    test_observation()
    test_observation_side_effects_and_plot()
    test_experiment_boundary_and_csv()
    test_public_teaching_entries()
    plt.close("all")
    print("RNN teaching tests passed on CPU.")


if __name__ == "__main__":
    main()
