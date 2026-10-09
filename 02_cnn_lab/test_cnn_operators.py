"""核对卷积/池化手算、尺寸与中文错误提示；不训练、不下载数据。"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cnn_lab.operators import show_convolution_step, show_pooling_step


def expect_error(action, text: str):
    try:
        action()
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError("无效参数应返回易懂的 ValueError。")


def check_numeric_steps():
    source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float64, requires_grad=True)
    before = source.detach().clone()
    weights = torch.tensor([[1, 0], [0, -1]], dtype=torch.float64)
    with redirect_stdout(StringIO()):
        step = show_convolution_step(source, weights, padding=1, stride=2, bias=0.5, output_position=(1, 1))
    assert step["input_shape"] == (1, 1, 3, 3)
    assert step["output_shape"] == (1, 1, 2, 2)
    torch.testing.assert_close(step["output"], torch.tensor([[-0.5, -2.5], [-6.5, -3.5]], dtype=torch.float64))
    torch.testing.assert_close(step["window"], torch.tensor([[5, 6], [8, 9]], dtype=torch.float64))
    torch.testing.assert_close(step["products"], torch.tensor([[5, 0], [0, -9]], dtype=torch.float64))
    assert step["product_sum"] == -4 and step["output_value"] == -3.5
    assert step["window_start"] == (2, 2)
    assert torch.equal(before, source.detach()) and source.grad is None
    assert not step["output"].requires_grad

    # 非方形输入的高、宽分别计算，而不是假定正方形。
    with redirect_stdout(StringIO()):
        rectangle = show_convolution_step([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], kernel_size=2)
    assert rectangle["output_shape"] == (1, 1, 2, 3)
    torch.testing.assert_close(rectangle["output"], torch.tensor([[14, 18, 22], [30, 34, 38]], dtype=torch.float64))

    image = [[1, 3, 2, 0], [4, 2, 1, 5], [0, 6, 2, 1], [3, 1, 4, 2]]
    with redirect_stdout(StringIO()):
        maximum = show_pooling_step(image, pooling="max", kernel_size=2, stride=2)
        average = show_pooling_step(image, pooling="avg", kernel_size=2, stride=2)
        overlap = show_pooling_step(image, pooling="avg", kernel_size=2, stride=1, output_position=(1, 1))
    torch.testing.assert_close(maximum["output"], torch.tensor([[4, 5], [6, 4]], dtype=torch.float64))
    torch.testing.assert_close(average["output"], torch.tensor([[2.5, 2], [2.5, 2.25]], dtype=torch.float64))
    assert overlap["output_shape"] == (1, 1, 3, 3)
    assert overlap["max_value"] == 6 and overlap["mean_value"] == 2.75
    assert overlap["output_value"] == 2.75
    assert maximum["max_value"] == 4 and maximum["mean_value"] == 2.5

    folder = Path(__file__).parent / "outputs" / "operator_lesson_qa"
    folder.mkdir(parents=True, exist_ok=True)
    for name, result in (("convolution_padding_stride", step), ("convolution_rectangle", rectangle),
                         ("pooling_max", maximum), ("pooling_average", average), ("pooling_overlap", overlap)):
        result["figure"].savefig(folder / f"{name}.png", dpi=140)
        plt.close(result["figure"])
    print(f"Numeric plots saved to {folder}")


def check_invalid_configs():
    image = [[1, 2], [3, 4]]
    cases = (
        (lambda: show_convolution_step(image, kernel_size=3), "放不进"),
        (lambda: show_convolution_step(image, kernel_size=0), "正整数"),
        (lambda: show_convolution_step(image, kernel_size=1, padding=-1), "非负整数"),
        (lambda: show_convolution_step(image, kernel_size=1, stride=0), "正整数"),
        (lambda: show_convolution_step(image, [[1, 2]], kernel_size=1), "方形"),
        (lambda: show_convolution_step(image, [[1]], kernel_size=2), "同步修改"),
        (lambda: show_convolution_step(image, kernel_size=1, bias=float("nan")), "有限"),
        (lambda: show_convolution_step([[float("inf")]], kernel_size=1), "无穷"),
        (lambda: show_convolution_step(image, kernel_size=1, output_position=(2, 0)), "超出"),
        (lambda: show_pooling_step(image, pooling="mean"), "请选择"),
        (lambda: show_pooling_step(image, kernel_size=3), "放不进"),
        (lambda: show_pooling_step(image, stride=0), "正整数"),
        (lambda: show_pooling_step([1, 2]), "二维"),
        (lambda: show_pooling_step(image, output_position=(-1, 0)), "非负整数"),
    )
    for action, message in cases:
        expect_error(action, message)


if __name__ == "__main__":
    torch.set_num_threads(1)
    check_numeric_steps()
    check_invalid_configs()
    print("CNN operator lesson checks passed.")
