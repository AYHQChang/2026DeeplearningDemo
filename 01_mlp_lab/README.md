# MLP 组件与决策边界实验室

本实验按“先看到结果，再理解代码”的顺序组织。第一次进入时，只需要打开
`00_一键体验.ipynb` 并点击 `Run All`。

## 推荐顺序

| 顺序 | Notebook | 主要任务 | 图片输出 |
|---:|---|---|---|
| 0 | `00_一键体验.ipynb` | 不改代码，完成第一次训练 | 训练阶段图、最终诊断图 |
| 1 | `01_认识数据与张量.ipynb` | 用图片、公式和代码理解数据集与 Tensor | 数据画廊、数据说明图、任务流程图 |
| 2 | `02_训练规则与超参数.ipynb` | 理解 Batch、Epoch、Loss、梯度和学习率 | Batch 覆盖图、学习率轨迹图 |
| 3 | `03_训练第一个MLP.ipynb` | 观察决策边界随 Epoch 形成 | 阶段图、Loss、Accuracy、梯度图 |
| 4 | `04_组件对比实验.ipynb` | 单独比较激活函数和优化器 | 两组组件对比图 |
| 5 | `05_自主修改练习.ipynb` | 每次只修改一个参数并记录结论 | 基线图、修改后诊断图 |

所有 notebook 均保存了一次成功运行的示例输出。即使尚未运行，也可以先查看结果图。

## Linux 中文图形

项目在 `assets/fonts` 中附带开源字体 Noto Sans CJK SC。绘图函数会自动注册该字体，
不需要在服务器中安装系统字体，也不需要修改每个账号的 Matplotlib 配置。

如果更新项目前已经打开 Notebook，请先执行 `git pull`，然后重启 Notebook 内核并
重新运行全部单元格。旧单元格中已经生成的方框图片不会自动刷新。

## 最短启动流程

1. 在 VS Code 中打开 `00_一键体验.ipynb`；
2. 选择课程使用的 Python 内核；
3. 点击 `Run All`；
4. 看到结果表、训练阶段图和最终诊断图后，再进入下一份 notebook。

Notebook 默认使用 `device="cpu"`，适合多人共享服务器。只有获得课堂分配后才改用
`device="cuda:x"`，并把 `x` 替换为指定的 GPU 编号，例如第 3 块写成 `cuda:3`。
不要使用 `auto` 抢占未分配的 GPU。

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
python demo.py --device cpu --mode fast --experiment baseline
python demo.py --device cpu --mode fast --experiment activation
```

## 最短检查

```bash
python test_mlp_smoke.py
```

看到 `MLP smoke test passed.` 表示数据、训练、绘图、简洁 API 和日志链路均可运行。

每份 Notebook 最后都有“清理并结束内核”单元格。运行它会清空变量、释放未使用的
CUDA 缓存，并正常结束当前 Python 进程。VS Code 随后显示内核停止或要求重新选择内核属于正常现象。
