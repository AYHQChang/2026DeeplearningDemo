# 环境安装与 Conda 常用命令

这份说明用于从零建立课程环境，并提供日常最常用的 Conda 命令。项目正式运行时完全离线；只有首次安装或更新环境需要联网。

## 1. 准备 Conda

电脑上安装 Anaconda 或 Miniconda，二者任选其一。安装完成后，打开 **Anaconda Prompt**，确认 Conda 可用：

```powershell
conda --version
```

如果能够显示版本号，就可以继续创建独立环境。不要把课程依赖直接安装到 `base` 环境，独立环境更容易复现，也不会影响其他项目。

## 2. 创建并进入 Ayitedu 环境

```powershell
conda create -n Ayitedu python=3.11 -y
conda activate Ayitedu
python --version
```

正常情况下，最后一条命令应显示 Python 3.11.x。以后每次运行 Demo 前，只需执行：

```powershell
conda activate Ayitedu
```

## 3. 安装 PyTorch：CPU 与 GPU 二选一

CPU 和 GPU 使用同一份项目代码，但 PyTorch 安装包不同。不要同时执行下面两条安装命令。

### 3.1 没有 NVIDIA GPU：安装 CPU 版本

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3.2 配有 NVIDIA GPU：安装 CUDA 版本

本课程当前开发环境使用 CUDA 12.8 对应的 PyTorch 安装包：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

不同显卡和不同时间可用的 CUDA 安装命令可能变化。需要在其他电脑安装 GPU 版本时，以 PyTorch 官方安装选择器给出的命令为准：<https://pytorch.org/get-started/locally/>。

安装完成后检查 PyTorch：

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

- CPU 版本显示 `CUDA available: False` 是正常现象；
- GPU 版本显示 `True` 时，可以使用 CUDA；
- 即使显示 `False`，三个 Demo 仍然可以强制使用 CPU 完整运行。

## 4. 安装其余依赖

先进入项目根目录，再安装两种设备共用的依赖：

```powershell
conda activate Ayitedu
cd D:\Code_for_all\2026DeeplearningDemo
python -m pip install -r requirements.txt
```

`requirements.txt` 没有包含 `torch`，目的是避免覆盖第 3 步已经选择好的 CPU 或 CUDA 版本。

建议始终使用 `python -m pip`，这样可以明确把依赖安装到当前激活环境对应的 Python 中。

## 5. 注册 Notebook 内核

注册一次后，VS Code 和 Jupyter 的内核列表中会出现 `Python (Ayitedu)`：

```powershell
python -m ipykernel install --user --name Ayitedu --display-name "Python (Ayitedu)"
```

打开 `01_mlp_lab/mlp_lab.ipynb` 后，选择 `Python (Ayitedu)`，再执行“Restart Kernel and Run All Cells”。

## 6. 验证课程环境

```powershell
conda activate Ayitedu
cd D:\Code_for_all\2026DeeplearningDemo
python check_environment.py
```

检查结果会列出 Python、PyTorch、NumPy、Matplotlib、scikit-learn、Jupyter 等版本，并显示 CUDA 是否可用。最后出现下面这行即表示核心环境通过：

```text
环境自检通过，可以离线运行课堂 Demo。
```

再运行一个最短示例：

```powershell
cd 01_mlp_lab
python demo.py --device cpu --mode fast --experiment baseline
```

## 7. CPU、GPU 与自动选择

同一份脚本通过 `--device` 切换设备：

```powershell
# 始终使用 CPU
python demo.py --device cpu --experiment baseline

# 有 CUDA 时用 GPU，否则自动使用 CPU
python demo.py --device auto --experiment baseline

# 必须使用 CUDA；CUDA 不可用时直接给出中文错误
python demo.py --device cuda --experiment baseline
```

Notebook 中修改顶部配置即可：

```python
DEVICE = "cpu"   # 或 "auto"、"cuda"
```

## 8. Conda 常用命令速查

### 8.1 查看与切换环境

| 目的 | 命令 |
|---|---|
| 查看 Conda 版本 | `conda --version` |
| 查看全部环境 | `conda env list` |
| 进入环境 | `conda activate Ayitedu` |
| 退出当前环境 | `conda deactivate` |
| 查看当前 Python | `python --version` |
| 查看当前 Python 位置 | `where python` |

### 8.2 创建、复制与删除环境

| 目的 | 命令 |
|---|---|
| 创建 Python 3.11 环境 | `conda create -n Ayitedu python=3.11 -y` |
| 复制现有环境 | `conda create -n Ayitedu_copy --clone Ayitedu` |
| 删除指定环境 | `conda remove -n Ayitedu_copy --all` |

删除环境会同时删除其中安装的全部包。执行前先用 `conda env list` 确认名称，并退出准备删除的环境。

### 8.3 查看、安装与更新包

| 目的 | 命令 |
|---|---|
| 查看当前环境全部包 | `conda list` |
| 查找某个包 | `conda list numpy` |
| 安装 Conda 包 | `conda install numpy` |
| 安装 requirements | `python -m pip install -r requirements.txt` |
| 查看 pip 包 | `python -m pip list` |
| 查看包的详细信息 | `python -m pip show torch` |
| 更新指定 Conda 包 | `conda update numpy` |
| 删除指定 Conda 包 | `conda remove numpy` |

同一个包不要反复交替使用 Conda 和 pip 安装。PyTorch 按本说明使用官方 pip 安装命令，其余课程依赖统一由 `requirements.txt` 管理。

### 8.4 导出与恢复环境记录

导出当前环境：

```powershell
conda env export -n Ayitedu > environment.yml
```

根据记录创建新环境：

```powershell
conda env create -f environment.yml
```

只记录主动安装的主要包，文件通常更简洁：

```powershell
conda env export -n Ayitedu --from-history > environment-history.yml
```

## 9. 每次使用前的最短流程

```powershell
conda activate Ayitedu
cd D:\Code_for_all\2026DeeplearningDemo\01_mlp_lab
python demo.py --device auto --mode fast --experiment baseline
```

使用 Notebook 时，将最后一行替换为：

```powershell
code mlp_lab.ipynb
```
