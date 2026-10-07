"""CNN 结构组合、数据迁移和最终测试边界的检查，不依赖 pytest。

在 02_cnn_lab 目录运行：python test_cnn_composition.py
这些小样本只检查功能，不用准确率证明教学效果或模型优劣。
"""

from contextlib import redirect_stdout
from dataclasses import replace
import csv
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from cnn_lab import (
    ComposableCNN, ConvBlock, ExperimentConfig, ImageDatasetBundle,
    evaluate_final, evaluation_data, experiment_records, load_digits_data, plot_digit_gallery,
    plot_feature_maps, plot_training_overview, print_model_summary, quick_demo,
    save_experiment_records, seed_everything, train_experiment, validate_config,
)
from cnn_lab import engine
from cnn_lab.data import validate_image_dataset


def expect_value_error(action, text: str = "") -> None:
    try:
        action()
    except ValueError as error:
        if text:
            assert text in str(error), str(error)
    else:
        raise AssertionError("无效配置或不符合输入/输出约定的模型应给出 ValueError。")


def toy_data(image_size: int = 28, classes: int = 3) -> ImageDatasetBundle:
    """每类均有样本的独立小划分，避免外部文件或下载。"""
    seed_everything(23)
    train_count, holdout_count = classes * 6, classes * 2
    return ImageDatasetBundle(
        x_train=torch.rand(train_count, 1, image_size, image_size),
        y_train=torch.arange(train_count, dtype=torch.long) % classes,
        x_val=torch.rand(holdout_count, 1, image_size, image_size),
        y_val=torch.arange(holdout_count, dtype=torch.long) % classes,
        x_test=torch.rand(holdout_count, 1, image_size, image_size),
        y_test=torch.arange(holdout_count, dtype=torch.long) % classes,
        class_names=tuple(f"class_{index}" for index in range(classes)),
        dataset_name=f"toy_{classes}classes_{image_size}", pixel_range=(0.0, 1.0),
    )


def check_structure_combinations() -> None:
    # 同一结构入口可处理数字 8×8 和 Fashion 28×28。
    # 每种深度均覆盖三种池化及训练中启用的 BN/Dropout。
    for image_size, classes in ((8, 10), (28, 3)):
        for depth in (1, 2, 3):
            for pooling in ("max", "avg", "none"):
                blocks = tuple(
                    ConvBlock(
                        out_channels=2 + index, kernel_size=(1, 3, 5)[index],
                        activation=("relu", "tanh", "leaky_relu")[index],
                        pooling=pooling, batch_norm=True, dropout=0.1,
                    )
                    for index in range(depth)
                )
                model = ComposableCNN(image_size, classes, blocks, dropout=0.1)
                inputs = torch.randn(4, 1, image_size, image_size)
                logits = model(inputs)
                assert logits.shape == (4, classes)
                loss = nn.CrossEntropyLoss()(logits, torch.arange(4) % classes)
                loss.backward()
                assert all(
                    parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters() if parameter.requires_grad
                )
                model.eval()
                maps = model.feature_maps(inputs)
                assert len(maps) == depth
                expected_size = image_size // (2 ** depth) if pooling != "none" else image_size
                assert list(maps.values())[-1].shape[-2:] == (expected_size, expected_size)

    # 结构预览不能更新 BN 统计或改变某个子模块原有的 train/eval 状态。
    model = ComposableCNN(8, 3, (ConvBlock(4, batch_norm=True, dropout=0.2),))
    model.train()
    model.classifier.eval()
    modes = {name: module.training for name, module in model.named_modules()}
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    with redirect_stdout(StringIO()):
        rows = print_model_summary(model, image_size=8)
    assert rows[-1]["output_shape"] == (1, 3)
    assert sum(row["parameters"] for row in rows) == sum(p.numel() for p in model.parameters())
    assert modes == {name: module.training for name, module in model.named_modules()}
    assert all(torch.equal(buffers[name], value) for name, value in model.named_buffers())

    for action in (
        lambda: ConvBlock(out_channels=0), lambda: ConvBlock(out_channels=65),
        lambda: ConvBlock(kernel_size=2), lambda: ConvBlock(kernel_size=9),
        lambda: ConvBlock(activation="unknown"), lambda: ConvBlock(pooling="unknown"),
        lambda: ConvBlock(batch_norm=1), lambda: ConvBlock(dropout=-0.1),
        lambda: ConvBlock(dropout=1.0),
        lambda: ComposableCNN(8, 3, ()),
        lambda: ComposableCNN(8, 3, (ConvBlock(),) * 4),
        lambda: ComposableCNN(8, 3, ({"out_channels": 4},)),
        lambda: ComposableCNN(4, 3, (ConvBlock(pooling="max"),) * 3),
    ):
        expect_value_error(action)


def check_validation_and_final_test():
    data = toy_data()
    config = ExperimentConfig(
        name="验证探索", blocks=(ConvBlock(3, pooling="avg"),),
        epochs=2, batch_size=6, learning_rate=0.001, seed=23,
    )
    original_accuracy = engine._accuracy

    def forbid_test(model, x, y, device):
        assert x is not data.x_test, "探索训练不应计算测试集准确率。"
        assert y is not data.y_test, "探索训练不应使用测试标签。"
        return original_accuracy(model, x, y, device)

    with patch("cnn_lab.engine._accuracy", side_effect=forbid_test):
        result = train_experiment(config, data=data, evaluate_test=False)
        # 同时检查图表中直接前向预测的输入：测试切片也共享原始 storage。
        test_storage = data.x_test.untyped_storage().data_ptr()

        def forbid_test_forward(_module, inputs):
            assert inputs[0].untyped_storage().data_ptr() != test_storage, (
                "探索诊断图应预测验证图片，不能提前预测测试图片。"
            )

        handle = result.model.register_forward_pre_hook(forbid_test_forward)
        try:
            figures = [plot_training_overview(result), plot_feature_maps(result, max_channels=2)]
            for figure in figures:
                buffer = BytesIO()
                figure.savefig(buffer, format="png")
                assert buffer.tell() > 0
        finally:
            handle.remove()
    assert result.data is data
    assert result.test_accuracy == []
    assert evaluation_data(result)[0] is data.x_val
    expect_value_error(lambda: evaluation_data(result, split="test"), "evaluate_final")
    assert len(result.val_accuracy) == config.epochs
    assert 0.0 <= result.final_validation_accuracy <= 1.0
    expect_value_error(lambda: result.final_test_accuracy, "evaluate_final")
    assert experiment_records([result])[0]["test_accuracy"] is None

    with patch("cnn_lab.engine._accuracy", wraps=original_accuracy) as accuracy:
        first = evaluate_final(result)
        assert 0.0 <= first <= 1.0 and result.test_accuracy == [first]
        assert accuracy.call_count == 1
        assert accuracy.call_args.args[1] is data.x_test
        assert evaluate_final(result) == first
        assert accuracy.call_count == 1, "重复最终评估应该返回保存成绩。"

    no_validation = replace(data, x_val=None, y_val=None)
    expect_value_error(
        lambda: train_experiment(config, data=no_validation, evaluate_test=False), "验证集",
    )
    return result


def check_custom_model_and_budgets() -> None:
    data = toy_data(image_size=8, classes=2)
    config = ExperimentConfig(name="自定义模型", epochs=1, batch_size=6, seed=23)
    seed_everything(config.seed)
    model = nn.Sequential(nn.Conv2d(1, 3, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = train_experiment(config, data=data, model=model, evaluate_test=False)
    assert result.model is model
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    wrong_classes = nn.Sequential(nn.Flatten(), nn.Linear(8 * 8, 3))
    expect_value_error(
        lambda: train_experiment(config, data=data, model=wrong_classes, evaluate_test=False), "输出",
    )
    wrong_size = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 2))
    expect_value_error(
        lambda: train_experiment(config, data=data, model=wrong_size, evaluate_test=False), "输入",
    )
    too_many_channels = nn.Sequential(nn.Conv2d(1, 65, 3), nn.Flatten(), nn.Linear(65 * 6 * 6, 2))
    expect_value_error(
        lambda: train_experiment(config, data=data, model=too_many_channels, evaluate_test=False), "64",
    )
    too_many_parameters = nn.Sequential(nn.Flatten(), nn.Linear(64, 12000), nn.Linear(12000, 2))
    expect_value_error(
        lambda: train_experiment(config, data=data, model=too_many_parameters, evaluate_test=False), "参数",
    )
    for unsafe in (
        replace(config, epochs=41), replace(config, batch_size=257),
        replace(config, blocks=()), replace(config, blocks=(ConvBlock(),) * 4),
    ):
        expect_value_error(lambda unsafe=unsafe: validate_config(unsafe))
    big_data = replace(
        data, x_train=data.x_train.repeat(20, 1, 1, 1), y_train=data.y_train.repeat(20),
    )
    expect_value_error(
        lambda: train_experiment(replace(config, epochs=40, batch_size=1), data=big_data, evaluate_test=False), "更新",
    )


def check_small_class_visuals_and_data_entry() -> None:
    for classes in (2, 3):
        data = toy_data(classes=classes)
        gallery = plot_digit_gallery(data)
        assert sum(bool(axis.images) for axis in gallery.axes) == classes
        for depth in (1, 3):
            config = ExperimentConfig(
                blocks=tuple(ConvBlock(2, pooling="none") for _ in range(depth)),
                epochs=1, batch_size=6,
            )
            result = train_experiment(config, data=data, evaluate_test=False)
            assert isinstance(result.model, ComposableCNN)
            features = plot_feature_maps(result, max_channels=2)
            assert sum(len(axis.images) for axis in features.axes) >= 2 * depth
    data = toy_data(classes=3)
    with patch("cnn_lab.api._show"), redirect_stdout(StringIO()):
        result = quick_demo(
            data=data, blocks=(ConvBlock(2, pooling="avg"),),
            epochs=1, batch_size=6, show_features=False, evaluate_test=False,
        )
    assert result.data is data
    assert result.model(data.x_val).shape == (len(data.x_val), 3)
    assert result.test_accuracy == []


def check_splits_and_records(selected) -> None:
    data = load_digits_data(seed=23, val_size=0.2)
    assert len(data.x_train) + len(data.x_val) + len(data.x_test) == 1797
    assert abs(len(data.x_val) / 1797 - 0.2) < 0.002
    assert data.x_train.shape[1:] == data.x_val.shape[1:] == data.x_test.shape[1:] == (1, 8, 8)
    repeat = load_digits_data(seed=23, val_size=0.2)
    assert torch.equal(data.x_train, repeat.x_train) and torch.equal(data.y_val, repeat.y_val)
    expect_value_error(lambda: load_digits_data(val_size=0.8, test_size=0.25))
    other = train_experiment(
        replace(selected.config, name="保留验证成绩"), data=selected.data, evaluate_test=False,
    )
    records = experiment_records([selected, other])
    assert records[0]["test_accuracy"] == selected.final_test_accuracy
    assert records[1]["test_accuracy"] is None
    assert records[0]["dataset"] == selected.data.dataset_name
    assert "C3" in records[0]["structure"]
    with TemporaryDirectory(prefix="cnn_records_", dir=Path(__file__).parent) as folder:
        path = Path(folder) / "nested" / "trials.csv"
        save_experiment_records([selected, other], path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == 2
        assert csv_rows[0]["name"] == selected.config.name
        assert csv_rows[1]["test_accuracy"] == ""
        assert csv_rows[1]["structure"] == records[1]["structure"]



def check_manual_dataset_validation() -> None:
    # 同一接口接受两类/三类和不同尺寸，不锁定Fashion或自动计分。
    for size, classes in ((8, 2), (28, 3), (16, 4)):
        data = toy_data(image_size=size, classes=classes)
        summary = validate_image_dataset(data)
        assert summary["image_size"] == size and len(summary["classes"]) == classes
        assert len(summary["splits"]["train"]["count_per_class"]) == classes
    data = toy_data()
    expect_value_error(lambda: validate_image_dataset(replace(data, x_val=None)), "val")
    expect_value_error(lambda: validate_image_dataset(replace(data, y_val=data.y_val[:-1])), "一一对应")
    expect_value_error(lambda: validate_image_dataset(replace(data, x_train=data.x_train.double())), "float32")
    expect_value_error(lambda: validate_image_dataset(replace(data, x_train=data.x_train * 255)), "归一化")
    expect_value_error(lambda: validate_image_dataset(replace(data, y_train=torch.zeros_like(data.y_train))), "每类")
    expect_value_error(lambda: validate_image_dataset(replace(data, x_val=data.x_val[:, :, :16, :16])), "尺寸")
    expect_value_error(lambda: validate_image_dataset(replace(data, class_names=("same", "same", "other"))), "类别名")


def main() -> None:
    seed_everything(23)
    check_structure_combinations()
    selected = check_validation_and_final_test()
    check_custom_model_and_budgets()
    check_small_class_visuals_and_data_entry()
    check_splits_and_records(selected)
    check_manual_dataset_validation()
    plt.close("all")
    print("CNN composition and dataset migration checks passed.")


if __name__ == "__main__":
    main()
