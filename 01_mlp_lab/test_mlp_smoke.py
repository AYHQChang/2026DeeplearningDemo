"""MLP 实验室的最小可运行烟雾测试。

这个文件不使用 pytest，直接执行即可完成以下检查：

1. CPU 设备解析、CUDA 错误提示和离线月牙数据的原始/标准化形状；
2. 一个 4-epoch 小模型能完成前向、反向、评价与阶段快照；
3. loss 数值有限、accuracy 范围合法、首末快照存在；
4. 阶段故事图和组件对比图能够在非交互 ``Agg`` 后端完成布局；
5. 损失函数对比确实生成 Cross-Entropy 与合法 MSE 两种配置；
6. 数据说明图、任务流程图和三种实验日志能够完成最小写入。
7. 自定义命令行配置的显示名称与实际结构和组件一致。

测试刻意使用小数据和少量 epoch，只验证代码链路，不把某次随机准确率当成固定
答案。完整的性能与可视化验收由 ``demo.py`` 和 Notebook 执行。
"""

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import (
    base_config,
    comparison_configs,
    configure_chinese_font,
    make_dataset,
    plot_comparison,
    plot_training_story,
    resolve_device,
    train_experiment,
)
from data_and_task import plot_data_explanation, plot_task_flow
from mlp_lab import quick_demo
from demo import build_parser, make_base_from_args, save_experiment_log


def main() -> None:
    """依次运行数据、训练、快照、绘图和配置断言。

    Returns:
        ``None``。全部断言通过时打印 ``MLP smoke test passed.``；任一断言失败
        会使进程返回非零退出码，适合在 PowerShell 或 VS Code Task 中调用。
    """

    assert resolve_device("cpu").type == "cpu"
    assert base_config().device == "cpu"
    assert build_parser().parse_args([]).device == "cpu"
    # 模拟没有 CUDA 的电脑，确认显式请求 GPU 时给出清楚的中文错误。
    with patch("mlp_lab.data.torch.cuda.is_available", return_value=False):
        try:
            resolve_device("cuda")
        except RuntimeError as error:
            assert "检测不到 CUDA" in str(error)
        else:
            raise AssertionError("CUDA 不可用时，resolve_device('cuda') 应报错。")
    with (
        patch("mlp_lab.data.torch.cuda.is_available", return_value=True),
        patch("mlp_lab.data.torch.cuda.device_count", return_value=2),
    ):
        assert resolve_device("cuda:1").index == 1
        try:
            resolve_device("cuda:2")
        except RuntimeError as error:
            assert "找不到第 2 块 GPU" in str(error)
        else:
            raise AssertionError("GPU 编号越界时应报错。")

    data = make_dataset("moons", n_samples=180, seed=7)
    assert data.x_train_raw.shape == data.x_train.shape
    assert data.x_test_raw.shape == data.x_test.shape
    assert data.x_train.shape[1] == 2
    assert data.n_classes == 2

    font_name = configure_chinese_font()
    assert font_name == "Noto Sans CJK SC"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        font_figure, font_axis = plt.subplots(figsize=(4, 2))
        font_axis.set_title("中文字体测试：训练与决策边界")
        font_axis.set_xlabel("标准化特征")
        font_buffer = BytesIO()
        font_figure.savefig(font_buffer, format="png")
        plt.close(font_figure)
    assert font_buffer.tell() > 0
    assert not any("Glyph" in str(item.message) for item in caught)

    config = replace(
        base_config(dataset="moons", mode="fast", seed=7, device="cpu"),
        epochs=4,
        n_samples=180,
        hidden_sizes=(8,),
    )
    result = train_experiment(config, data=data)
    assert torch.get_num_threads() == 1
    assert len(result.train_loss) == config.epochs
    assert all(np.isfinite(result.train_loss))
    assert 0.0 <= result.final_test_accuracy <= 1.0
    assert {0, config.epochs}.issubset(result.snapshots)

    story = plot_training_story(result)
    comparison = plot_comparison([result], "烟雾测试")
    data_figure = plot_data_explanation([data])
    task_figure = plot_task_flow(data.dataset_name, data.n_classes)
    assert len(story.axes) >= 8
    assert len(comparison.axes) >= 4
    assert len(data_figure.axes) == 4
    assert len(task_figure.axes) == 1
    plt.close("all")

    with TemporaryDirectory() as temporary_dir:
        json_path, text_path, csv_path = save_experiment_log(
            [result], "smoke", "cpu", "cpu", "fast", Path(temporary_dir)
        )
        assert json_path.exists() and text_path.exists() and csv_path.exists()
        assert "final_test_accuracy" in json_path.read_text(encoding="utf-8")

    losses = comparison_configs("loss", config)
    assert {item.loss_name for item in losses} == {"cross_entropy", "mse"}

    api_result = quick_demo(
        dataset="moons", epochs=2, n_samples=90, device="cpu", show_story=False
    )
    assert 0.0 <= api_result.final_test_accuracy <= 1.0
    plt.close("all")

    custom_args = build_parser().parse_args(
        ["--hidden-sizes", "8,8", "--activation", "tanh", "--optimizer", "sgd"]
    )
    custom_config = make_base_from_args(custom_args)
    assert all(value in custom_config.name for value in ("8x8", "tanh", "sgd"))
    print("MLP smoke test passed.")


if __name__ == "__main__":
    main()
