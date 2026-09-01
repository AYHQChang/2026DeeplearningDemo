# MLP 组件与决策边界实验室

这是一个可独立运行的二维分类实验。建议只沿下面这条主线开始：

1. 阅读 [`快速上手与练习指南.md`](快速上手与练习指南.md)；
2. 打开 `mlp_lab.ipynb`；
3. 选择 CPU 或 GPU，执行 `Restart Kernel and Run All Cells`；
4. 完成 Notebook 中的分级练习，并填写实验记录。

## CPU / GPU

两种设备共用同一套代码，只修改 Notebook 顶部的 `DEVICE`：

```python
DEVICE = "cpu"   # CPU
DEVICE = "cuda"  # NVIDIA GPU
```

命令行入口使用同样的选择方式：

```powershell
python demo.py --device cpu --mode fast --experiment baseline
python demo.py --device cuda --mode fast --experiment baseline
```

## 文件职责

| 文件 | 用途 | 是否需要首先阅读 |
|---|---|---|
| `快速上手与练习指南.md` | 启动流程、修改地图、分级任务和实验记录 | 是 |
| `mlp_lab.ipynb` | 演示、对照实验和动手练习的主入口 | 是 |
| `core.py` | 数据、模型、组件、训练与绘图的唯一实现 | 按修改地图选读 |
| `demo.py` | PowerShell / VS Code 命令行入口 | 可选 |
| `data_and_task.py` | 训练前的数据与任务说明图 | 可选 |
| `使用手册.md` | 参数、日志、绘图和命令速查 | 遇到问题时查阅 |
| `test_mlp_smoke.py` | 最小链路检查 | 维护时使用 |

## 最短检查

```powershell
python test_mlp_smoke.py
```

看到 `MLP smoke test passed.` 表示数据、训练、快照、绘图和日志链路均可运行。
