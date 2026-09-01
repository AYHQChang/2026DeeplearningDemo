# 深度学习课堂 Demo

本仓库用于“讲练结合”的深度学习课堂。当前按“一次完成一个 Demo”的原则，先实现：

- `01_mlp_lab`：MLP 组件选择与二维决策边界实验室。

CNN 与 RNN 实验室将在 MLP Demo 验收后再逐个实现。

## 环境

从零创建环境、选择 CPU/GPU 版 PyTorch、注册 Notebook 内核及 Conda 命令速查见：

- `环境安装与Conda常用命令.md`

本机已指定使用 Conda 环境 `Ayitedu`。PyTorch 的 CPU/GPU 安装包不同，先按课堂电脑选择一种：

```powershell
conda activate Ayitedu

# 无 NVIDIA GPU 的电脑：
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 配有 RTX 5070 Ti 的电脑：
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

然后安装两种设备共用的教学依赖：

```powershell
python -m pip install -r requirements.txt
```

之所以不把 `torch` 写入 `requirements.txt`，是为了防止后一步安装错误地覆盖已经选好的 CPU/CUDA 版本。

正式课堂运行不需要联网，也不会下载数据或模型。

## 快速开始

```powershell
cd 01_mlp_lab
# 先看数据、任务、输入输出和默认基线：
python data_and_task.py

# 再开始训练；每次训练默认自动保存日志：
python demo.py --device cpu --mode fast --experiment baseline
```

也可以打开 `01_mlp_lab/mlp_lab.ipynb`，按顺序运行全部单元格。

完整的参数、CPU/GPU 切换和绘图位置见 `01_mlp_lab/使用手册.md`。
