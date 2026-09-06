"""房价 Notebook 的教学绘图模块。

本文件只把数据和训练结果转换为图片，不生成数据、不训练模型。每个函数
返回 Matplotlib Figure，由 Notebook 决定何时显示；烟雾测试也能在无界面
服务器上保存并检查图片。所有图先配置项目自带中文字体。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from house_price import FEATURES, price_metrics
from mlp_lab.plots import configure_chinese_font


def house_table(x, y, title="训练集中的前 8 套模拟房源", rows=8):
    """将原始房源 [N,4] 与价格排成表格，帮助区分输入 X 和目标 y。"""
    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 3), layout="constrained")
    # 数据可能少于 rows 行，因此只显示实际存在的数量。
    count = min(rows, len(x))
    # table 接收字符串矩阵；统一保留一位小数使表格更紧凑。
    values = [[f"{v:.1f}" for v in row] + [f"{p:.1f}"]
              for row, p in zip(x[:count], np.asarray(y).reshape(-1)[:count])]
    table = ax.table(cellText=values, colLabels=[*FEATURES, "价格（万元）"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    # 四个输入表头为蓝色，目标价格表头为黄色。
    for j in range(5):
        table[0, j].set_facecolor("#DBEAFE" if j < 4 else "#FDE68A")
    ax.axis("off")
    ax.set_title(title, pad=15)
    return fig


def plot_data(data):
    """展示训练集四个特征、价格分布和三个子集的数量。

    前四图一次只观察一个特征与价格，适合建立直觉，但没有控制其他特征，
    因而不能用来证明因果关系。
    """
    configure_chinese_font()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    # 探索图只读取训练集，不提前观察验证集或测试集关系。
    x, y = data["x_train_raw"], data["y_train_raw"].ravel()
    # 2×3 画布的前四格依次对应输入矩阵 X 的四列。
    for j, ax in enumerate(axes.flat[:4]):
        ax.scatter(x[:, j], y, s=15, alpha=0.5, color="#2563EB")
        ax.set(xlabel=FEATURES[j], ylabel="总价（万元）", title=f"{FEATURES[j]} 与价格")
        ax.grid(alpha=0.2)
    # 第五格用于识别模型主要见过的价格范围。
    axes[1, 1].hist(y, bins=20, color="#60A5FA", edgecolor="white")
    axes[1, 1].set(xlabel="总价（万元）", ylabel="房源数量", title="训练集价格分布")
    # 第六格核对 60/20/20 划分。
    counts = [len(data["indices"][name]) for name in ("train", "val", "test")]
    bars = axes[1, 2].bar(["训练", "验证", "测试"], counts, color=["#2563EB", "#F59E0B", "#94A3B8"])
    axes[1, 2].bar_label(bars)
    axes[1, 2].set(title="60% / 20% / 20%：各有任务", ylabel="房源数量", ylim=(0, max(counts) * 1.2))
    fig.suptitle("观察训练数据：面积不是决定价格的唯一输入（模拟数据）", fontsize=14)
    return fig


def plot_scaling(data):
    """并排展示同五套房源标准化前后的数值。

    左图保留平方米、年、公里等原始单位，右图使用训练集统计量转换。
    两个面板中的同一行仍是同一套房源。
    """
    configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    # 五行足以看清变化，同时避免数字过密。
    for ax, values, title in zip(axes, [data["x_train_raw"][:5], data["x_train"][:5].numpy()],
                                  ["原始单位：数值尺度不同", "标准化后：仍是相同的 5 套房源"]):
        ax.imshow(values, aspect="auto", cmap="Blues")
        # 在色块上写数值；深色格使用白字以保持可读性。
        for i in range(5):
            for j in range(4):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center",
                        color="white" if values[i, j] > (values.min() + values.max()) / 2 else "black")
        ax.set_xticks(range(4), ["面积", "房间数", "房龄", "距离"])
        ax.set_yticks(range(5), [f"样本 {i}" for i in range(5)])
        ax.set_title(title)
    return fig


def plot_losses():
    """画出 MSE、MAE、Huber 对同一标准化预测误差的惩罚曲线。"""
    configure_chinese_font()
    # 这些点只用于画损失函数形状，不参与模型训练。
    error = torch.linspace(-3, 3, 200)
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    # reduction="none" 保留每个误差点的损失，才能连成曲线。
    for name, criterion in [("MSE", nn.MSELoss(reduction="none")),
                             ("MAE", nn.L1Loss(reduction="none")),
                             ("Huber (delta=1)", nn.HuberLoss(reduction="none"))]:
        ax.plot(error, criterion(error, torch.zeros_like(error)), label=name)
    ax.set(xlabel="标准化预测误差", ylabel="单样本损失", title="同样的误差，不同损失如何惩罚？")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def plot_training(result):
    """画训练/验证曲线，并标出最终恢复的最佳验证轮次。

    左图采用当前损失函数的标准化尺度；右图统一采用万元 MAE，适合在
    不同损失函数之间比较。
    """
    configure_chinese_font()
    h = result["history"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), layout="constrained")
    # 同图叠加训练集与验证集，便于观察欠拟合或过拟合迹象。
    for prefix, label in [("train", "训练"), ("val", "验证")]:
        axes[0].plot(h["epoch"], h[f"{prefix}_loss"], label=label)
        axes[1].plot(h["epoch"], h[f"{prefix}_mae"], label=label)
    axes[0].set(ylabel="标准化价格上的 Loss", title=f"当前损失：{result['config']['loss']}")
    axes[1].set(ylabel="MAE（万元）", title="还原单位后：平均差多少钱？")
    # 两个面板使用相同的最佳轮次标记与阅读辅助元素。
    for ax in axes:
        ax.axvline(result["best_epoch"], color="gray", ls=":", label="所选轮次")
        ax.set_xlabel("Epoch（0 表示尚未训练）")
        ax.legend()
        ax.grid(alpha=0.2)
    return fig


def plot_snapshots(result, data):
    """展示多个 Epoch 的验证集真实价格与预测价格。

    每个点是一套验证房源；越接近对角线，预测越接近真实值。全部面板
    共用坐标范围，避免缩放差异造成某轮看起来更好的错觉。
    """
    configure_chinese_font()
    snapshots = result["snapshots"]
    truth = data["y_val_raw"].ravel()
    # 综合真实值和所有阶段预测确定统一范围，并留出 10 万元边距。
    all_prices = np.concatenate([truth, *[p.ravel() for p in snapshots.values()]])
    lo, hi = all_prices.min() - 10, all_prices.max() + 10
    fig, axes = plt.subplots(1, len(snapshots), figsize=(15, 3.5), squeeze=False, layout="constrained")
    # 字典保持插入顺序，面板按训练函数保存的 Epoch 顺序排列。
    for ax, (epoch, prediction) in zip(axes.flat, snapshots.items()):
        ax.scatter(truth, prediction.ravel(), s=13, alpha=0.6)
        ax.plot([lo, hi], [lo, hi], "--", color="gray")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="真实价（万元）",
               ylabel="预测价（万元）", title=f"Epoch {epoch}")
    fig.suptitle("同一验证集：点越接近虚线，预测越准确；所有面板共用刻度")
    return fig


def plot_comparison(results):
    """用统一的验证 MAE 比较多组训练配置。

    左图展示收敛过程，右图展示每组最佳验证轮次的 MAE。不同训练损失
    的原始 Loss 尺度不同，所以这里全部换成万元 MAE。
    """
    configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    # results 的键直接成为图例，值是 train_model 返回的结果字典。
    for label, result in results.items():
        axes[0].plot(result["history"]["epoch"], result["history"]["val_mae"], label=label)
    axes[0].set(xlabel="Epoch", ylabel="验证 MAE（万元）", title="用同一种指标比较实验")
    axes[0].legend()
    # 柱高是每组的最佳验证 MAE，不是最后一个 Epoch 的 MAE。
    bars = axes[1].bar(list(results), [r["val_mae"] for r in results.values()], color="#60A5FA")
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].margins(y=0.2)
    axes[1].tick_params(axis="x", labelrotation=15)
    axes[1].set(ylabel="验证 MAE（万元）", title="各实验所选轮次的验证误差")
    return fig


def plot_predictions(truth, prediction, title="测试集预测"):
    """从三个角度诊断一组万元价格预测。

    第一图检查预测是否接近对角线；第二图观察残差是否随预测价格系统
    变化；第三图并排比较前八套房的真实价与预测价。总标题汇总 MAE、
    RMSE 和 R²。
    """
    configure_chinese_font()
    # 展平后，每个真实价格与同位置预测价格一一对应。
    truth, prediction = np.asarray(truth).ravel(), np.asarray(prediction).ravel()
    metrics = price_metrics(truth, prediction)
    # 预测减真实：正值表示高估，负值表示低估。
    error = prediction - truth
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    # 对角线覆盖真实值和预测值的共同范围。
    lo, hi = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
    axes[0].scatter(truth, prediction, s=20, alpha=0.65)
    axes[0].plot([lo, hi], [lo, hi], "--", color="gray")
    axes[0].set(xlabel="真实价（万元）", ylabel="预测价（万元）", title="虚线上方高估，下方低估")
    axes[1].scatter(prediction, error, s=20, alpha=0.65)
    axes[1].axhline(0, color="gray", ls="--")
    axes[1].set(xlabel="预测价（万元）", ylabel="预测 − 真实（万元）", title="误差是否随价格改变？")
    # 固定取前八套，不按误差挑选，避免只展示效果好的样本。
    index = np.arange(min(8, len(truth)))
    axes[2].bar(index - 0.18, truth[:len(index)], width=0.36, label="真实价")
    axes[2].bar(index + 0.18, prediction[:len(index)], width=0.36, label="预测价")
    axes[2].set(xlabel="前 8 套房源（不挑选最佳样本）", ylabel="总价（万元）", xticks=index)
    axes[2].legend()
    fig.suptitle(f"{title}｜MAE {metrics['MAE']:.2f} 万元｜RMSE {metrics['RMSE']:.2f} 万元｜R² {metrics['R2']:.3f}")
    return fig
