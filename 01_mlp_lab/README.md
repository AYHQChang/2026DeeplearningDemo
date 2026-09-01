# MLP 组件与决策边界实验室

本实验按“先看到结果，再理解代码”的顺序组织。第一次进入时，只需要打开
`00_一键体验.ipynb` 并点击 `Run All`。

## 推荐顺序

| 顺序 | Notebook | 主要任务 | 图片输出 |
|---:|---|---|---|
| 0 | `00_一键体验.ipynb` | 不改代码，完成第一次训练 | 训练阶段图、最终诊断图 |
| 1 | `01_认识数据与张量.ipynb` | 认识数据、训练/测试划分和张量形状 | 数据画廊、数据说明图 |
| 2 | `02_训练第一个MLP.ipynb` | 观察决策边界随 Epoch 形成 | 阶段图、Loss、Accuracy、梯度图 |
| 3 | `03_组件对比实验.ipynb` | 单独比较激活函数和优化器 | 两组组件对比图 |
| 4 | `04_自主修改练习.ipynb` | 每次只修改一个参数并记录结论 | 基线图、修改后诊断图 |

所有 notebook 均保存了一次成功运行的示例输出。即使尚未运行，也可以先查看结果图。

## 最短启动流程

1. 在 VS Code 中打开 `00_一键体验.ipynb`；
2. 选择课程使用的 Python 内核；
3. 点击 `Run All`；
4. 看到结果表、训练阶段图和最终诊断图后，再进入下一份 notebook。

Notebook 默认使用 `device="auto"`：检测到可用 CUDA 时使用 GPU，否则自动使用 CPU。
如果需要固定设备，可以改为 `device="cpu"` 或 `device="cuda"`。

## 代码结构

| 文件 | 用途 | 是否需要首先阅读 |
|---|---|---|
| `mlp_lab/api.py` | `quick_demo()`、`compare()` 等简洁入口 | 否，直接调用即可 |
| `mlp_lab/model.py` | MLP 与激活函数，约几十行 | 完成参数练习后再读 |
| `mlp_lab/data.py` | 数据生成、划分、标准化和设备选择 | 否 |
| `mlp_lab/engine.py` | 实验配置、训练循环和单变量对比 | 否 |
| `mlp_lab/plots.py` | 决策边界与训练诊断图 | 否 |
| `core.py` | 兼容旧脚本的导入入口 | 不需要阅读 |
| `demo.py` | VS Code / 终端命令行入口 | 可选 |
| `data_and_task.py` | 数据与任务说明图 | 可选 |
| `使用手册.md` | 参数、日志、绘图和命令速查 | 遇到问题时查询 |

## 最短代码

```python
from mlp_lab import quick_demo

result = quick_demo()
```

组件对比：

```python
from mlp_lab import compare

results = compare("activation")
```

## 命令行入口

```bash
python demo.py --device auto --mode fast --experiment baseline
python demo.py --device auto --mode fast --experiment activation
```

## 最短检查

```bash
python test_mlp_smoke.py
```

看到 `MLP smoke test passed.` 表示数据、训练、绘图、简洁 API 和日志链路均可运行。
