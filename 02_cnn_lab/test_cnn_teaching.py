"""逐层观察功能的 CPU 检查，不依赖 pytest、外部图片或训练成绩。

在 02_cnn_lab 目录运行：python test_cnn_teaching.py
检查真实前向结果、状态恢复和图中的数据，而不把特征图当作解释正确的证据。
"""

from contextlib import redirect_stdout
from io import BytesIO, StringIO
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cnn_lab import (
    ComposableCNN, ConvBlock, SmallCNN, plot_forward_pass,
    print_model_summary, trace_forward,
)


def state_snapshot(model):
    return (
        {name: module.training for name, module in model.named_modules()},
        {name: tensor.detach().clone() for name, tensor in model.named_buffers()},
        {name: (len(module._forward_hooks), len(module._forward_pre_hooks))
         for name, module in model.named_modules()},
    )


def assert_state_unchanged(model, snapshot):
    modes, buffers, hooks = snapshot
    assert modes == {name: module.training for name, module in model.named_modules()}
    assert hooks == {
        name: (len(module._forward_hooks), len(module._forward_pre_hooks))
        for name, module in model.named_modules()
    }
    assert all(torch.equal(buffers[name], tensor) for name, tensor in model.named_buffers())


def inspect_and_compare(model, images):
    """独立挂钩记录真实操作，避免只核对 helper 自己生成的文字与形状。"""
    actual = []
    handles = []

    def record(module, inputs, output):
        actual.append((
            type(module).__name__, tuple(inputs[0].shape),
            output.detach().cpu().clone(),
        ))

    before = state_snapshot(model)
    try:
        for module in model.modules():
            if not list(module.children()):
                handles.append(module.register_forward_hook(record))
        rows = trace_forward(model, images)
    finally:
        for handle in handles:
            handle.remove()
    assert_state_unchanged(model, before)
    assert len(rows) == len(actual) + 1, "应按实际执行顺序记录输入及每次叶子操作。"
    assert tuple(rows[0]["output_shape"]) == tuple(images.shape)
    torch.testing.assert_close(rows[0]["tensor"], images.detach().cpu())
    for row, (kind, input_shape, output) in zip(rows[1:], actual):
        assert {"layer", "label", "type", "input_shape", "output_shape",
                "parameters", "details", "tensor"}.issubset(row)
        assert row["type"] == kind
        assert tuple(row["input_shape"]) == input_shape
        assert tuple(row["output_shape"]) == tuple(output.shape)
        assert row["tensor"].device.type == "cpu" and not row["tensor"].requires_grad
        torch.testing.assert_close(row["tensor"], output)
    assert sum(row["parameters"] for row in rows) == sum(p.numel() for p in model.parameters())
    return rows


def check_baseline_and_combinations():
    images = torch.rand(2, 1, 8, 8)
    model = SmallCNN(8)
    rows = inspect_and_compare(model, images)
    convolutions = [row for row in rows if row["type"] == "Conv2d"]
    assert [row["output_shape"] for row in convolutions] == [(2, 8, 8, 8), (2, 16, 4, 4)]
    pools = [row for row in rows if row["type"] == "MaxPool2d"]
    assert len(pools) == 1 and pools[0]["output_shape"] == (2, 8, 4, 4)
    flat = [row for row in rows if row["type"] == "Flatten"]
    assert flat[0]["output_shape"] == (2, 256)
    assert rows[-1]["output_shape"] == (2, 10)
    # 基础模型的同一个激活模块调用两次，不能只显示一次。
    activations = [row for row in rows if "Activation" in row["type"] or row["type"] == "ReLU"]
    assert len(activations) == 2
    # 保留旧 Notebook 对 feature_maps 的接口，不把原有调用变成细分操作列表。
    assert list(model.feature_maps(images)) == ["第一层卷积", "池化后", "第二层卷积"]

    for size, classes in ((8, 10), (28, 3), (16, 2)):
        for depth in (1, 2, 3):
            for pooling in ("max", "avg", "none"):
                blocks = tuple(
                    ConvBlock(
                        out_channels=index + 2, kernel_size=(1, 3, 5)[index],
                        activation=("relu", "tanh", "leaky_relu")[index],
                        pooling=pooling, batch_norm=True, dropout=0.2,
                    ) for index in range(depth)
                )
                model = ComposableCNN(size, classes, blocks, dropout=0.1)
                model.train()
                model.classifier.eval()
                rows = inspect_and_compare(model, torch.rand(2, 1, size, size))
                assert rows[-1]["output_shape"] == (2, classes)
                expected_size = size // (2 ** depth) if pooling != "none" else size
                last_conv = [row for row in rows if row["type"] == "Conv2d"][-1]
                pre_last_pool_size = (size // (2 ** (depth - 1))) if pooling != "none" else size
                assert last_conv["output_shape"][-2:] == (pre_last_pool_size,) * 2
                flat = next(row for row in rows if row["type"] == "Flatten")
                assert flat["output_shape"] == (2, (depth + 1) * expected_size ** 2)
                pool_rows = [row for row in rows if row["type"] in {"MaxPool2d", "AvgPool2d"}]
                assert len(pool_rows) == (0 if pooling == "none" else depth)


def check_summary_and_shared_modules():
    model = ComposableCNN(16, 2, (ConvBlock(3, batch_norm=True, pooling="avg"),))
    model.train()
    model.blocks[0].activation.eval()
    before = state_snapshot(model)
    with redirect_stdout(StringIO()) as output:
        rows = print_model_summary(model, image_size=16)
    assert rows[-1]["output_shape"] == (1, 2)
    assert sum(row["parameters"] for row in rows) == sum(p.numel() for p in model.parameters())
    assert output.getvalue().strip()
    assert_state_unchanged(model, before)

    class SharedConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)
            self.relu = nn.ReLU()
            self.flat = nn.Flatten()
            self.output = nn.Linear(64, 2)

        def forward(self, x):
            x = self.relu(self.conv(x))
            x = self.relu(self.conv(x))
            return self.output(self.flat(x))

    shared = SharedConv()
    rows = inspect_and_compare(shared, torch.rand(2, 1, 8, 8))
    conv_rows = [row for row in rows if row["type"] == "Conv2d"]
    assert len(conv_rows) == 2
    assert sum(row["parameters"] for row in conv_rows) == sum(p.numel() for p in shared.conv.parameters())

    class FunctionalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3, padding=1)
            self.output = nn.Linear(128, 2)

        def forward(self, x):
            # 任意自定义函数运算不伪造为模块；其形状变化可由相邻输入输出检查。
            return self.output(F.relu(self.conv(x)).flatten(1))

    functional = FunctionalModel()
    rows = inspect_and_compare(functional, torch.rand(2, 1, 8, 8))
    assert [row["type"] for row in rows[1:]] == ["Conv2d", "Linear"]
    assert rows[1]["output_shape"] == (2, 2, 8, 8)
    assert rows[2]["input_shape"] == (2, 128)


def check_failure_and_gradients():
    class BrokenForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3, padding=1)
            self.bn = nn.BatchNorm2d(2)

        def forward(self, x):
            self.bn(self.conv(x))
            raise RuntimeError("教学测试中的故意失败")

    broken = BrokenForward()
    broken.train()
    broken.conv.eval()
    existing_handle = broken.conv.register_forward_hook(lambda *_args: None)
    try:
        before = state_snapshot(broken)
        for action in (
            lambda: trace_forward(broken, torch.rand(1, 1, 8, 8)),
            lambda: print_model_summary(broken, image_size=8),
        ):
            try:
                action()
            except RuntimeError as error:
                assert "故意失败" in str(error)
            else:
                raise AssertionError("检查工具不应吞掉实际前向传播错误。")
            assert_state_unchanged(broken, before)
    finally:
        existing_handle.remove()

    model = ComposableCNN(8, 3, (ConvBlock(3, batch_norm=True, pooling="max", dropout=0.2),))
    model.train()
    model.classifier.eval()
    images = torch.rand(2, 1, 8, 8, requires_grad=True)
    rows = trace_forward(model, images)
    input_copy = rows[0]["tensor"].clone()
    with torch.no_grad():
        images.add_(0.1)
    assert torch.equal(rows[0]["tensor"], input_copy), "返回的张量必须与调用者输入分离。"
    loss = nn.CrossEntropyLoss()(model(images), torch.tensor([0, 1]))
    loss.backward()
    assert images.grad is not None and bool(torch.isfinite(images.grad).all())
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters() if parameter.requires_grad)


def check_plot_data_and_fonts():
    for size, classes, depth in ((8, 10, 2), (28, 3, 3), (16, 2, 1)):
        model = (SmallCNN(size, classes) if size == 8 else ComposableCNN(
            size, classes, tuple(ConvBlock(index + 2, pooling="avg", batch_norm=True)
                                 for index in range(depth)),
        ))
        model.train()
        model.classifier.eval()
        images = torch.rand(2, 1, size, size)
        rows = trace_forward(model, images[:1])
        before = state_snapshot(model)
        with redirect_stdout(StringIO()):
            figures = plot_forward_pass(
                model, images, class_names=tuple(f"类别{i}" for i in range(classes)), max_channels=2,
            )
        assert len(figures) == 2
        feature, head = figures
        assert_state_unchanged(model, before)
        # 图中必须含真实卷积及激活张量，不能只画每个完整卷积块的最终输出。
        arrays = [np.asarray(image.get_array()) for axis in feature.axes for image in axis.images]
        for row in rows:
            tensor = row["tensor"]
            if tensor.ndim == 4:
                for channel in range(min(2, tensor.shape[1])):
                    wanted = tensor[0, channel].numpy()
                    assert any(array.shape == wanted.shape and np.allclose(array, wanted)
                               for array in arrays), row["label"]
        # 分类头图必须有真实 logits 或概率数值；布局可采用曲线、柱形或图片。
        logits = rows[-1]["tensor"][0].numpy()
        probabilities = torch.softmax(rows[-1]["tensor"], dim=1)[0].numpy()
        vectors = [np.asarray(line.get_ydata()) for axis in head.axes for line in axis.lines]
        vectors += [np.asarray([patch.get_height() for patch in axis.patches])
                    for axis in head.axes if axis.patches]
        vectors += [np.asarray(image.get_array()).reshape(-1)
                    for axis in head.axes for image in axis.images]
        assert any(vector.shape == logits.shape and np.allclose(vector, logits) for vector in vectors)
        assert any(vector.shape == probabilities.shape and np.allclose(vector, probabilities) for vector in vectors)
        for figure in figures:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                buffer = BytesIO()
                figure.savefig(buffer, format="png", dpi=75)
                assert buffer.tell() > 1000
            missing = [str(item.message) for item in caught if "Glyph" in str(item.message)]
            assert not missing, "中文/符号缺字：" + "; ".join(missing)
            plt.close(figure)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    check_baseline_and_combinations()
    check_summary_and_shared_modules()
    check_failure_and_gradients()
    check_plot_data_and_fonts()
    plt.close("all")
    print("CNN teaching trace, state preservation, and figure data checks passed.")


if __name__ == "__main__":
    main()
