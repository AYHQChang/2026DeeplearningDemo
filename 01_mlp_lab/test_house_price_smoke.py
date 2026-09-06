"""房价回归实验的最小烟雾测试。

运行：python 01_mlp_lab/test_house_price_smoke.py

本文件不依赖 pytest 或网络。它用小数据检查“数据 → 模型 → 训练 →
预测 → 绘图”整条链路。断言失败时进程返回非零状态；全部通过时打印
House price smoke test passed. 具体误差不是固定答案。
"""

from io import BytesIO
from itertools import product
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from house_price import make_houses, prepare_data, build_model, train_model, predict_prices, price_metrics
from house_price_plots import house_table, plot_data, plot_scaling, plot_losses, plot_training, plot_snapshots, plot_comparison, plot_predictions


def main():
    """检查数据隔离、组件组合、预测还原、错误提示和图片渲染。"""
    # 150 条数据足以检查流程，同时让测试保持快速。
    x, y = make_houses(150)
    data = prepare_data(x, y)

    # 1. 三个子集必须互不重叠，并完整覆盖全部原始行。
    index = data["indices"]
    assert not set(index["train"]) & set(index["val"])
    assert not set(index["test"]) & (set(index["train"]) | set(index["val"]))
    assert sorted(np.concatenate(list(index.values()))) == list(range(len(x)))
    # 2. 标准化器均值必须来自训练集，不能使用完整数据。
    np.testing.assert_allclose(data["x_scaler"].mean_, x[index["train"]].mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(data["y_scaler"].mean_, y[index["train"]].mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(data["y_scaler"].inverse_transform(data["y_train"]), y[index["train"]], rtol=1e-6)

    # 3. 删去测试字段后仍能训练，证明 train_model 不读取测试答案。
    train_val = {k: v for k, v in data.items() if "test" not in k}
    train_val["indices"] = {k: v for k, v in index.items() if k != "test"}
    # 3 种损失 × 3 种优化器全部完成一次短训练。
    for loss, optimizer in product(("mse", "mae", "huber"), ("adam", "sgd", "rmsprop")):
        result = train_model(train_val, loss=loss, optimizer=optimizer, epochs=2, batch_size=17)
        assert np.isfinite(result["val_mae"])
        assert result["model"](data["x_val"]).shape == data["y_val"].shape
    # 每种激活函数都应保持回归输出形状 [B, 1]。
    for activation in ("relu", "tanh", "sigmoid"):
        assert build_model((8,), activation)(torch.zeros(5, 4)).shape == (5, 1)

    # 4. 相同数据、配置和种子重复训练，应得到相同验证曲线。
    result = train_model(train_val, epochs=50)
    repeated = train_model(train_val, epochs=50)
    np.testing.assert_allclose(result["history"]["val_mae"], repeated["history"]["val_mae"])
    # 5. 预测还原为万元后，重新计算的 MAE 应与训练结果一致。
    val_prediction = predict_prices(result, data, data["x_val_raw"])
    assert abs(price_metrics(data["y_val_raw"], val_prediction)["MAE"] - result["val_mae"]) < 1e-4
    assert result["best_epoch"] == 1 + np.argmin(result["history"]["val_mae"][1:])
    # 模型应优于“所有房都猜训练均价”的最低教学基线。
    baseline = np.full_like(data["y_val_raw"], data["y_scaler"].mean_[0])
    assert result["val_mae"] < price_metrics(data["y_val_raw"], baseline)["MAE"]
    assert next(result["model"].parameters()).device.type == "cpu"
    assert torch.get_num_threads() == 1
    np.testing.assert_allclose(predict_prices(result, data, [[100, 3, 8, 5]]).shape, (1, 1))
    # 超出训练范围的新房源在开启检查时应产生外推提醒。
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predict_prices(result, data, [[1000, 3, 8, 5]], check_range=True)
    assert any("外推" in str(w.message) for w in caught)

    # 6. 常见无效配置应尽早抛出 ValueError。
    for change in ({"loss": "cross_entropy"}, {"optimizer": "bad"}, {"lr": 0}, {"epochs": 0}):
        try:
            train_model(train_val, **change)
        except ValueError:
            pass
        else:
            raise AssertionError(f"无效配置未拒绝：{change}")
    # 缺失值也应在数据入口被拒绝，避免训练后出现 nan Loss。
    bad = x.copy()
    bad[0, 0] = np.nan
    try:
        prepare_data(bad, y)
    except ValueError:
        pass
    else:
        raise AssertionError("缺失数据未拒绝")

    # 7. 使用无界面的 Agg 后端真正保存图片，并检查中文字体警告。
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figures = [house_table(x, y), plot_data(data), plot_scaling(data), plot_losses(),
                   plot_training(result), plot_snapshots(result, data),
                   plot_comparison({"基线": result}), plot_predictions(data["y_val_raw"], val_prediction)]
        for figure in figures:
            # 写入内存，不在项目目录中留下测试图片。
            buffer = BytesIO()
            figure.savefig(buffer, format="png")
            assert buffer.tell() > 1000
            plt.close(figure)
    assert not any("Glyph" in str(w.message) for w in caught)
    print("House price smoke test passed.")


if __name__ == "__main__":
    # 导入本模块不会自动训练；直接执行文件时才运行测试。
    main()
