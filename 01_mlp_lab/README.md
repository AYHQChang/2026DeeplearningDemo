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
| 6 | `06_MLP闯关项目.ipynb` | 四小时综合闯关、最小代码修改与证据报告 | 三星计分、诊断图、实验表 |

前六份 notebook 保存了一次成功运行的示例输出；闯关项目不预填结果，留给学生形成自己的证据链。

## 四小时综合项目

`06_MLP闯关项目.ipynb` 把工作量分配给地图观察、实验预测、三组单变量对照、
阅读并修改 `model.py`、预算内调参和 300–500 字证据报告，共 240 分钟。正式训练
最多 9 次，每次保持 CPU、小数据和少量 Epoch；四小时来自分析深度，不来自长时间占用服务器。

闯关计分同时检查 Accuracy、参数量和更新次数，不把运行时间计入分数。训练配置还设置了
共享服务器安全上限：`epochs≤200`、`n_samples≤1500`、`batch_size≤512`、隐藏层≤4 层且
每层≤128；单次实验预计更新次数不能超过 12000。

## 1.1：房价预测，认识回归

基础 Notebook 之后，可以继续运行 [11_房价预测.ipynb](./11_房价预测.ipynb)。推荐已读过数据与张量、训练规则，并至少训练过一次分类 MLP 后再进入。

- 学习路线：模拟房源表 → 特征散点图 → 标准化与 Tensor → MLP 训练 → 单变量比较 → 测试集 → 输入自己的房源。
- 参数集中在 `CONFIG`，对比项在 `EXPERIMENTS`，自主练习在 `MY_CHANGE`，待预测房源在 `HOUSE`。
- 损失可选 `mse / mae / huber`，优化器可选 `adam / sgd / rmsprop`；也能修改隐藏层、激活、学习率、batch_size、epochs 和 weight_decay。
- 默认 CPU 单线程，无新增依赖、无数据下载；图中金额统一为万元，数据仅为教学模拟。
- 验证集负责选参数和轮次，测试集留到最后。不同损失使用统一的万元 MAE 比较。
- 末尾附同格式 CSV 替换示例、练习记录模板及关闭内核代码。

`house_price.py` 保存数据、模型、训练与预测；`house_price_plots.py` 保存绘图排版。它们为回归单独组织教学代码，不改变原来的分类 API。

从项目根目录检查：

```bash
python 01_mlp_lab/test_house_price_smoke.py
```

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
| `mlp_lab/api.py` | `quick_demo()`、`compare()`、`challenge_report()` 等简洁入口 | 否，直接调用即可 |
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
