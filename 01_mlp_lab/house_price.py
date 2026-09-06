"""房价回归实验的计算模块。

本文件按“生成数据 → 划分与标准化 → 搭建 MLP → 训练与选模 → 预测”
组织代码。建议按 make_houses、prepare_data、build_model、train_model、
predict_prices 的顺序阅读。

绘图位于 house_price_plots.py。模拟价格公式只负责生成教学数据，不会
传给模型，也不代表真实房产市场。
"""

from copy import deepcopy
import warnings

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlp_lab.data import resolve_device

# 输入矩阵 X 的四列必须始终保持这个顺序。
FEATURES = ("面积（平方米）", "房间数", "房龄（年）", "距中心（公里）")

# Notebook 可覆盖这些默认值。集中保存配置，便于检查参数并公平比较实验。
DEFAULT_CONFIG = dict(hidden_sizes=(32, 16), activation="relu", loss="mse",
                      optimizer="adam", lr=0.01, epochs=120, batch_size=32,
                      weight_decay=0.0, seed=42, device="cpu")


def make_houses(n_samples=600, seed=42):
    """生成可重复的模拟房源。

    参数：
        n_samples：房源数量。
        seed：随机种子；相同种子会得到相同房源。

    返回：
        x：float32 输入矩阵 [N, 4]，列顺序见 FEATURES。
        price：float32 目标 [N, 1]，单位为万元。
    """
    if n_samples < 30:
        raise ValueError("请至少生成 30 条房源，便于划分三个数据子集。")
    # 局部随机数生成器不会改变程序其他位置的随机状态。
    rng = np.random.default_rng(seed)
    area = rng.uniform(40, 180, n_samples)
    # 房间数与面积有关，但加入扰动后不会形成简单的一一对应。
    rooms = np.clip(np.rint(area / 35 + rng.normal(0, 0.5, n_samples)), 1, 5)
    age = rng.uniform(0, 35, n_samples)
    distance = rng.uniform(0.5, 20, n_samples)
    # 四个长度为 N 的数组按列拼成 [N, 4]。
    x = np.column_stack((area, rooms, age, distance)).astype(np.float32)
    # 规则含线性项、非线性项和交互项；噪声表示四个特征外的价格差异。
    price = (20 + 1.2 * area + 8 * rooms - 0.65 * age - 1.5 * distance
             + 0.003 * (area - 90) ** 2
             + 0.04 * area * np.maximum(8 - distance, 0)
             + rng.normal(0, 8, n_samples))
    # 模型输出 [B, 1]，所以目标也保存成二维列向量 [N, 1]。
    return x, price.astype(np.float32).reshape(-1, 1)


def prepare_data(x, y, seed=42):
    """检查数据，按 60/20/20 划分，并标准化输入与价格。

    x 要求为 [N, 4]；y 可为 [N] 或 [N, 1]，内部统一成 [N, 1]。
    返回字典同时保存原始数组、标准化 Tensor、子集行号和两个标准化器。
    所有 fit 只使用训练集，防止验证集或测试集信息泄漏。
    """
    # float32 与 PyTorch 默认模型参数类型一致，也更节省内存。
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    # 接受常见的一维目标写法，内部统一为列向量。
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    # 在数据入口检查形状和有限值，让错误在训练前出现。
    if x.ndim != 2 or x.shape[1] != 4 or y.shape != (len(x), 1) or len(x) < 30:
        raise ValueError("要求 X=[N,4]、y=[N,1]，N 至少为 30；特征顺序见 FEATURES。")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("数据含缺失值或无穷值，请先清理数据。")
    # 先留下 40%，再平分为验证集和测试集；划分行号便于检查互不重叠。
    train, remaining = train_test_split(np.arange(len(x)), test_size=0.4, random_state=seed)
    val, test = train_test_split(remaining, test_size=0.5, random_state=seed)
    # sx 处理四个输入特征，sy 处理一个价格；二者只查看训练集。
    sx = StandardScaler().fit(x[train])
    sy = StandardScaler().fit(y[train])
    data = dict(x_scaler=sx, y_scaler=sy, indices=dict(train=train, val=val, test=test))
    # 原始值用于画图和万元评价，标准化 Tensor 用于送入 MLP。
    for name, index in data["indices"].items():
        data[f"x_{name}_raw"], data[f"y_{name}_raw"] = x[index].copy(), y[index].copy()
        data[f"x_{name}"] = torch.tensor(sx.transform(x[index]), dtype=torch.float32)
        data[f"y_{name}"] = torch.tensor(sy.transform(y[index]), dtype=torch.float32)
    return data


def build_model(hidden_sizes=(32, 16), activation="relu"):
    """搭建 4 → 隐藏层 → 1 的全连接回归网络。

    例如 hidden_sizes=(32, 16) 得到 4 → 32 → 16 → 1。每个隐藏层后
    加激活函数；末层直接输出一个连续值，不加 Softmax 或 argmax。
    """
    # 将 Notebook 中便于修改的短名称映射到 PyTorch 层类型。
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
    if activation not in activations or any(not isinstance(w, int) or w <= 0 for w in hidden_sizes):
        raise ValueError("activation 选 relu/tanh/sigmoid；隐藏层宽度必须为正整数。")
    # width 记录上一层宽度，第一层固定接收四个房源特征。
    layers, width = [], 4
    for next_width in hidden_sizes:
        layers.extend([nn.Linear(width, next_width), activations[activation]()])
        width = next_width
    # 房价回归每个样本只输出一个数，末层后不再添加激活函数。
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def price_metrics(truth, prediction):
    """计算原始万元尺度的 MAE、RMSE 和 R²。

    [N] 与 [N, 1] 输入都会先展平。MAE、RMSE 的单位为万元；R²
    没有单位，模型比猜评价集均值更差时可以为负。
    """
    # 每个真实价格必须与同位置的预测价格一一对应。
    truth, prediction = np.asarray(truth).reshape(-1), np.asarray(prediction).reshape(-1)
    if truth.shape != prediction.shape or len(truth) == 0:
        raise ValueError("真实值和预测值必须一一对应且非空。")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("评价数据含非有限数值。")
    # 正误差表示高估，负误差表示低估。
    error = prediction - truth
    # 真实值完全相同时 R² 分母为 0，返回 nan 表示该指标无定义。
    total = np.sum((truth - truth.mean()) ** 2)
    return dict(MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt(np.mean(error ** 2))),
                R2=float(1 - np.sum(error ** 2) / total) if total > 0 else float("nan"))


def train_model(data, **changes):
    """训练一组配置，并返回最佳验证轮次的模型。

    changes 只覆盖 DEFAULT_CONFIG 中已有的键。训练集负责更新权重，验证集
    负责选择轮次，本函数完全不读取测试集。

    返回的字典包含 CPU 模型、完整配置、逐轮历史、阶段预测、最佳轮次
    和该轮的验证 MAE。
    """
    # 先拒绝拼错的参数名，避免看似成功、实际修改却没有生效。
    unknown = changes.keys() - DEFAULT_CONFIG.keys()
    if unknown:
        raise ValueError(f"未知参数：{sorted(unknown)}")
    # 默认配置与本次修改合并，Notebook 可以只传想改变的项目。
    config = {**DEFAULT_CONFIG, **changes}
    # 将便于阅读的字符串选项转换为真正的 PyTorch 类。
    losses = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.HuberLoss}
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
    # 创建模型前检查配置，使错误信息更直接。
    if config["loss"] not in losses or config["optimizer"] not in optimizers:
        raise ValueError("loss 选 mse/mae/huber；optimizer 选 adam/sgd/rmsprop。")
    if any(not isinstance(config[k], int) or config[k] <= 0 for k in ("epochs", "batch_size")):
        raise ValueError("epochs 和 batch_size 必须为正整数。")
    if not np.isfinite(config["lr"]) or config["lr"] <= 0 or not np.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("lr 必须为有限正数，weight_decay 必须为有限非负数。")
    # 共享课堂只占一个 CPU 线程；固定种子使单变量比较可复现。
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    # resolve_device 会检查 cuda:x 是否存在；课堂默认使用 CPU。
    device = resolve_device(config["device"])
    model = build_model(config["hidden_sizes"], config["activation"]).to(device)
    criterion = losses[config["loss"]]()
    optimizer = optimizers[config["optimizer"]](model.parameters(), lr=config["lr"],
                                               weight_decay=config["weight_decay"])
    # 每轮打乱训练样本。专用 generator 保证批次顺序可重复；
    # num_workers=0 表示不为每个 Notebook 再创建数据加载子进程。
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(TensorDataset(data["x_train"], data["y_train"]),
                        batch_size=config["batch_size"], shuffle=True, generator=generator,
                        num_workers=0)
    # 训练集和验证集移到目标设备；测试集不会进入本函数。
    x_train, y_train = data["x_train"].to(device), data["y_train"].to(device)
    x_val, y_val = data["x_val"].to(device), data["y_val"].to(device)
    # history 用于曲线，snapshots 用于观察预测随 Epoch 的变化。
    history = {key: [] for key in ("epoch", "train_loss", "val_loss", "train_mae", "val_mae")}
    snapshots, best_mae, best_state, best_epoch = {}, float("inf"), None, 0
    milestones = {0, 1, 10, 40, config["epochs"]}

    # 多循环一次以记录 Epoch 0，即随机初始化、尚未更新的模型。
    for epoch in range(config["epochs"] + 1):
        if epoch > 0:
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)  # 二者形状分别为 [B,4]、[B,1]
                optimizer.zero_grad()                 # 清除上一批的梯度
                prediction = model(xb)                # 前向：预测标准化价格 [B,1]
                loss = criterion(prediction, yb)      # 比较预测和目标
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss 非有限值：可先降低学习率并检查数据尺度。")
                loss.backward()                       # 反向：计算梯度
                optimizer.step()                      # 按梯度更新参数

        # 每轮结束后统一评价。eval 和 no_grad 不更新参数，也不保留梯度图。
        model.eval()
        with torch.no_grad():
            train_pred, val_pred = model(x_train), model(x_val)
            train_loss, val_loss = criterion(train_pred, y_train).item(), criterion(val_pred, y_val).item()
        # 网络输出位于标准化尺度；先还原成万元，才能计算直观的 MAE。
        val_price = data["y_scaler"].inverse_transform(val_pred.cpu().numpy())
        train_price = data["y_scaler"].inverse_transform(train_pred.cpu().numpy())
        train_mae = price_metrics(data["y_train_raw"], train_price)["MAE"]
        val_mae = price_metrics(data["y_val_raw"], val_price)["MAE"]
        # 依照 history 的键顺序追加本轮五项记录。
        for key, value in zip(history, (epoch, train_loss, val_loss, train_mae, val_mae)):
            history[key].append(value)
        # 只保存阶段图需要的轮次，避免复制每一轮的完整预测。
        if epoch in milestones:
            snapshots[epoch] = val_price.copy()
        # Epoch 0 用于展示未训练状态；候选模型从第 1 轮开始。
        # deepcopy 保存独立参数副本，不让它随后续训练继续变化。
        if epoch > 0 and val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch
            best_state = deepcopy(model.state_dict())

    # 恢复验证 MAE 最小轮次，而不是直接使用最后一轮参数。
    model.load_state_dict(best_state)
    # 返回 CPU 模型便于画图与预测，也避免多组实验一直占用 GPU。
    model.cpu().eval()
    return dict(model=model, config=config, history=history, snapshots=snapshots,
                best_epoch=best_epoch, val_mae=best_mae)


def predict_prices(result, data, houses, check_range=False):
    """预测一批原始单位房源，并返回万元价格 [N, 1]。

    houses 必须为 [N, 4]，单位和列顺序与训练数据一致。函数复用训练时
    的标准化器；check_range=True 时，超出训练范围会给出外推提醒。
    """
    # 统一成训练数据所用的 float32 NumPy 格式。
    houses = np.asarray(houses, dtype=np.float32)
    if houses.ndim != 2 or houses.shape[1] != 4 or len(houses) == 0 or not np.isfinite(houses).all():
        raise ValueError("请提供有限数值组成的二维房源数组 [N,4]。")
    # 范围检查只发出提醒，不阻止课堂演示外推。
    train = data["x_train_raw"]
    if check_range and np.any((houses < train.min(axis=0)) | (houses > train.max(axis=0))):
        warnings.warn("部分输入超出训练集特征范围，属于外推，预测可能不可靠。", stacklevel=2)
    # 必须复用训练阶段的 x_scaler；重新 fit 会改变模型看到的特征含义。
    x = torch.tensor(data["x_scaler"].transform(houses), dtype=torch.float32)
    # 返回模型位于 CPU；预测不需要梯度。
    with torch.no_grad():
        standardized = result["model"](x).numpy()
    # 最后把标准化输出恢复成 Notebook 展示的万元单位。
    return data["y_scaler"].inverse_transform(standardized)
