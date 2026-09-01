"""CNN 实验室的最小可运行检查，不依赖 pytest。"""

from io import BytesIO
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cnn_lab import (
    SmallCNN, base_config, compare_pooling_experiments, configure_chinese_font,
    load_digits_data, plot_convolution_demo, plot_digit_gallery,
    plot_feature_maps, plot_pixel_and_shape, plot_pooling_comparison,
    plot_pooling_demo, plot_training_overview, resolve_device, shift_images_right,
    train_experiment,
)


def main() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert base_config().device == "cpu"
    with patch("cnn_lab.data.torch.cuda.is_available", return_value=False):
        try:
            resolve_device("cuda")
        except RuntimeError as error:
            assert "检测不到 CUDA" in str(error)
        else:
            raise AssertionError("CUDA 不可用时应报错。")

    data = load_digits_data(seed=7)
    assert data.x_train.ndim == 4
    assert data.x_train.shape[1:] == (1, 8, 8)
    assert data.image_size == 8 and data.n_classes == 10
    assert 0.0 <= float(data.x_train.min()) <= float(data.x_train.max()) <= 1.0

    # 扩展护栏：以后换成 28×28 数据时，模型和分类层不需要重写。
    future_model = SmallCNN(image_size=28, n_classes=10)
    assert future_model(torch.randn(2, 1, 28, 28)).shape == (2, 10)

    marker = torch.zeros(1, 1, 3, 4)
    marker[..., 0] = 1
    shifted = shift_images_right(marker, 1)
    assert shifted[..., 0].sum() == 0 and shifted[..., 1].sum() == 3

    config = base_config(seed=7)
    config = type(config)(**{**config.__dict__, "channels": (4, 8), "epochs": 2})
    result = train_experiment(config, data=data)
    assert len(result.train_loss) == 2
    assert 0.0 <= result.final_test_accuracy <= 1.0
    assert torch.get_num_threads() == 1
    maps = result.model.feature_maps(data.x_test[:2])
    assert maps["第一层卷积"].shape == (2, 4, 8, 8)
    assert maps["第二层卷积"].shape == (2, 8, 4, 4)

    configure_chinese_font()
    figures = [
        plot_digit_gallery(data), plot_pixel_and_shape(data),
        plot_convolution_demo(data), plot_pooling_demo(data),
        plot_training_overview(result), plot_feature_maps(result, max_channels=4),
    ]
    comparison_base = type(config)(**{**config.__dict__, "channels": (2, 4), "epochs": 1})
    comparison = compare_pooling_experiments(comparison_base, data=data)
    figures.append(plot_pooling_comparison(comparison))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        buffer = BytesIO()
        figures[0].savefig(buffer, format="png")
    assert buffer.tell() > 0
    assert not any("Glyph" in str(item.message) for item in caught)
    plt.close("all")
    print("CNN smoke test passed.")


if __name__ == "__main__":
    main()
