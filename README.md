# 🧠 深度学习课堂 Demo

> 不从一大段代码开始，而是先点一下、看到结果，再一步步理解模型为什么会这样表现。

欢迎来到这个“讲一点、跑一下、改一处、看一张图”的深度学习实验项目！🎉

这里的目标不是背代码，也不是只追求更高的 Accuracy，而是通过可以直接运行的 Notebook，真正看懂数据、模型、训练过程和实验对比。

## 🎯 这个项目是做什么的？

每个实验室都会先把一个深度学习主题拆成六个基础步骤；完成综合建设的实验室再增加一份约四小时的观察、编程和证据项目：

```text
一键看到结果
      ↓
认识数据与张量
      ↓
理解训练规则与超参数
      ↓
训练第一个模型
      ↓
只改变一个组件进行对比
      ↓
自己修改参数或少量代码
```

你不需要先读完很长的训练代码。第一次进入时，只要打开第一份 Notebook，选择 Python 内核并点击 `Run All`，就能看到图片和结果。

## ✅ 当前已经可以学习什么？

### `01_mlp_lab`：MLP 与二维决策边界

这是当前已经完成并经过运行检查的实验室。

你会使用月牙、同心圆和螺旋等二维数据，观察一个 MLP 怎样逐渐形成决策边界，并比较激活函数、优化器、网络结构、学习率和 Dropout 等组件。

| 顺序 | Notebook | 你会看到什么 |
|---:|---|---|
| 0 | [00_一键体验.ipynb](./01_mlp_lab/00_一键体验.ipynb) | 一次完整训练、决策边界和诊断图 |
| 1 | [01_认识数据与张量.ipynb](./01_mlp_lab/01_认识数据与张量.ipynb) | 数据形状、标签、训练集与测试集 |
| 2 | [02_训练规则与超参数.ipynb](./01_mlp_lab/02_训练规则与超参数.ipynb) | Batch、Epoch、Loss、梯度和学习率 |
| 3 | [03_训练第一个MLP.ipynb](./01_mlp_lab/03_训练第一个MLP.ipynb) | 决策边界如何随 Epoch 逐渐形成 |
| 4 | [04_组件对比实验.ipynb](./01_mlp_lab/04_组件对比实验.ipynb) | 激活函数和优化器的单变量对比 |
| 5 | [05_自主修改练习.ipynb](./01_mlp_lab/05_自主修改练习.ipynb) | 修改一个参数后，与基线结果进行比较 |
| 6 | [06_MLP闯关项目.ipynb](./01_mlp_lab/06_MLP闯关项目.ipynb) | 四小时综合闯关、代码修改与证据报告 |

前六份 Notebook 保存了成功运行的图形输出；第 7 份是待填写的综合项目。
即使暂时没有配置好运行环境，也可以先在 GitHub 上打开前六份预览实验结果。👀

更详细的 MLP 使用说明见 [01_mlp_lab/README.md](./01_mlp_lab/README.md)。

### `1.1`：用 MLP 预测房价 🏠

完成 MLP 基础后，打开 [11_房价预测.ipynb](./01_mlp_lab/11_房价预测.ipynb)，把“判断类别”换成“预测一个连续数值”。

先看模拟房源表和散点图，再看模型怎样学会预测价格。你可以自己输入面积、房间数、房龄和距离，也可以在参数区切换 MSE / MAE / Huber 损失、Adam / SGD / RMSprop 优化器，修改学习率、Batch、Epoch 和隐藏层宽度。

本页还会解释训练集、验证集、测试集各自的任务，用万元尺度的 MAE 比较实验，并与“始终猜训练均价”的简单基线比较。数据完全离线生成，明确标注为模拟数据；不用于真实房产估价。

### `02_cnn_lab`：CNN 与小型图像分类

这是第二个已经完成并经过运行检查的实验室。第一版使用离线的 `8×8` 手写数字数据，让卷积核、特征图和池化结果更容易直接看清；代码同时保留了 `28×28` 输入接口，后续可以自然扩展到更完整的手写数字任务。

| 顺序 | Notebook | 你会看到什么 |
|---:|---|---|
| 0 | [00_一键体验.ipynb](./02_cnn_lab/00_一键体验.ipynb) | 一次完整 CNN 训练、预测和诊断图 |
| 1 | [01_图片数据与四维张量.ipynb](./02_cnn_lab/01_图片数据与四维张量.ipynb) | 图片、像素值与 `[N, C, H, W]` 四维张量 |
| 2 | [02_卷积核与池化.ipynb](./02_cnn_lab/02_卷积核与池化.ipynb) | 卷积核响应、最大池化和平均池化 |
| 3 | [03_训练第一个CNN.ipynb](./02_cnn_lab/03_训练第一个CNN.ipynb) | Loss、Accuracy、混淆矩阵、错误样本与特征图 |
| 4 | [04_CNN组件对比实验.ipynb](./02_cnn_lab/04_CNN组件对比实验.ipynb) | 不同池化方式的单变量对比 |
| 5 | [05_自主修改练习.ipynb](./02_cnn_lab/05_自主修改练习.ipynb) | 修改卷积通道数，并与基线结果比较 |
| 6 | [06_CNN识别侦探项目.ipynb](./02_cnn_lab/06_CNN识别侦探项目.ipynb) | 四小时综合取证、代码修改与预算挑战 |

更详细的 CNN 使用说明见 [02_cnn_lab/README.md](./02_cnn_lab/README.md)。

### `03_rnn_lab`：RNN、LSTM、GRU 与长期记忆

这个实验室把序列学习变成一个直观的“记秘密”任务：序列开头给出红、绿或蓝，经过一段与答案无关的噪声后，模型要在末尾回答最初的颜色。数据由本地代码生成，无需下载。

| 顺序 | Notebook | 你会看到什么 |
|---:|---|---|
| 0 | [00_一键体验.ipynb](./03_rnn_lab/00_一键体验.ipynb) | 秘密时间线、训练曲线、混淆矩阵和预测概率 |
| 1 | [01_认识序列与三维张量.ipynb](./03_rnn_lab/01_认识序列与三维张量.ipynb) | `[Batch, Time, Feature]` 与实际数据怎样对应 |
| 2 | [02_隐藏状态与循环记忆.ipynb](./03_rnn_lab/02_隐藏状态与循环记忆.ipynb) | 循环展开图、隐藏状态热力图和输入梯度 |
| 3 | [03_训练第一个RNN.ipynb](./03_rnn_lab/03_训练第一个RNN.ipynb) | Vanilla RNN 的完整训练、诊断和预测案例 |
| 4 | [04_RNN_LSTM_GRU对比.ipynb](./03_rnn_lab/04_RNN_LSTM_GRU对比.ipynb) | 不同序列长度下的准确率、参数量和训练时间 |
| 5 | [05_自主修改练习.ipynb](./03_rnn_lab/05_自主修改练习.ipynb) | 保留基线、只改一项并记录证据 |

核心实验默认使用 CPU。更详细的运行方法见 [03_rnn_lab/README.md](./03_rnn_lab/README.md)。

### `04_transformer_lab`：小型 Transformer 与注意力寻宝

这个实验室不用大语言模型和在线语料，而是让一层 Transformer 从数量相同的红、绿、蓝 token 中找回第 0 位颜色。实验中可以直接观察位置编码、Q/K/V、Attention 权重和组件对比。

| 顺序 | Notebook | 你会看到什么 |
|---:|---|---|
| 0 | [00_一键体验.ipynb](./04_transformer_lab/00_一键体验.ipynb) | 彩色 token、训练曲线、混淆矩阵和 QUERY Attention |
| 1 | [01_Token与三维张量.ipynb](./04_transformer_lab/01_Token与三维张量.ipynb) | `[B,T]` token ids 到 `[B,T,D]` Embedding |
| 2 | [02_位置编码与QKV.ipynb](./04_transformer_lab/02_位置编码与QKV.ipynb) | 正弦位置编码和一次 Scaled Dot-product Attention 手算 |
| 3 | [03_训练第一个Transformer.ipynb](./04_transformer_lab/03_训练第一个Transformer.ipynb) | 一层 Encoder 的训练、梯度、预测和 Attention |
| 4 | [04_Transformer组件对比.ipynb](./04_transformer_lab/04_Transformer组件对比.ipynb) | 位置编码开关与 1/2/4 Head 受控对比 |
| 5 | [05_自主修改练习.ipynb](./04_transformer_lab/05_自主修改练习.ipynb) | 修改一行激活映射并保留证据 |
| 6 | [06_注意力寻宝项目.ipynb](./04_transformer_lab/06_注意力寻宝项目.ipynb) | 240 分钟综合项目、四徽章和预算挑战 |

共享入口默认 CPU 单线程并拒绝 `device="auto"`；模型、序列、数据量、更新次数、参数量和 Attention 组合预算均有训练前硬上限。更详细的运行方法见 [04_transformer_lab/README.md](./04_transformer_lab/README.md)。

## 🔍 运行一次实验，可以观察哪些信息？

- 输入数据和 tensor shape；
- 训练集与测试集的划分；
- 模型参数量和训练时间；
- 决策边界的形成过程；
- Loss 与 Accuracy 曲线；
- 误判样本出现在哪里；
- Gradient norm 是否稳定；
- 修改前后的数值证据和图形证据。

重点不是记住某次运行的数字，而是学会回答：

> 我只修改了什么？结果发生了什么变化？哪张图能够支持我的判断？这个结论在什么条件下成立？

## 🌱 完成这些实验后，你将能够

- 在 VS Code 中打开和运行 Jupyter Notebook；
- 看懂常见 tensor shape，区分特征与标签；
- 理解训练集、测试集和基本的数据标准化；
- 区分 Batch、Batch size、Iteration 和 Epoch；
- 理解 Loss、Gradient 与 Learning rate 如何共同完成一次参数更新；
- 认识 PyTorch 模型训练的完整流程；
- 用图形观察模型如何学习，而不是只看最终准确率；
- 使用单变量对照比较不同模型组件；
- 在 CPU 或 GPU 上运行同一份实验代码；
- 修改参数，并逐步过渡到修改少量模型代码；
- 根据数值和图片写出有证据、有边界的实验结论。

## ⚡ CPU / GPU 速度教学入口

四个项目都保留各自的 `demo.py`。需要公平展示 CPU 与 GPU 训练速度时，统一从仓库根目录运行：

```bash
python device_speed_demo.py --lab mlp --devices cpu,cuda:0
python device_speed_demo.py --lab cnn --devices cpu,cuda:0
python device_speed_demo.py --lab rnn --devices cpu,cuda:0
python device_speed_demo.py --lab transformer --devices cpu,cuda:0 --preview-epochs 2
```

把 `cuda:0` 换成教师分配的 GPU 编号。脚本始终顺序运行两个设备，不使用 `auto`，也不一次启动四个项目。

`--target-epochs` 表示完整训练目标，`--preview-epochs` 表示实际只运行多少个 Epoch。预览模式会输出训练时间、每 Epoch 时间、端到端时间、预计完整时间、预计剩余时间、GPU 峰值显存和 CPU/GPU 加速比。估时是按当前每 Epoch 线性外推的粗略值；预览 Accuracy 不作为最终效果结论。

这些课堂模型很小，GPU 可能因为初始化、数据传输和小算子开销而比 CPU 更慢。这是需要解释的实验结果，不是程序故障。运行时间只用于认识设备与计算规模，不计入课程成绩。

## 🚀 最快开始方式

### 第一步：获取项目

```bash
git clone https://github.com/AYHQChang/2026DeeplearningDemo.git
cd 2026DeeplearningDemo
```

### 第二步：进入课程环境并检查

Windows 自带电脑第一次配置时，先阅读 [Windows 自带电脑：Conda 与 PyTorch 安装指南](./5.Windows自带电脑Conda与PyTorch安装.md)，并运行只读预检：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\check_windows_environment.ps1"
```

预检只报告电脑、Conda、Python、PyTorch 和 NVIDIA 驱动状态，不会自动安装或修改系统。根据指南选择 CPU 或 NVIDIA CUDA 路线后，再执行下面的统一验收。

```bash
conda activate dl2026
python check_environment.py
python 01_mlp_lab/test_mlp_smoke.py
python 01_mlp_lab/test_house_price_smoke.py
python 02_cnn_lab/test_cnn_smoke.py
python 03_rnn_lab/test_rnn_smoke.py
python 04_transformer_lab/test_transformer_smoke.py
python test_device_speed_demo.py
```

看到以下内容，说明基本环境、四个实验室和统一速度入口可以运行：

```text
环境自检通过，可以离线运行课堂 Demo。
MLP smoke test passed.
House price smoke test passed.
CNN smoke test passed.
RNN smoke test passed.
Transformer smoke test passed.
Device speed demo smoke test passed.
```

### 第三步：打开第一份 Notebook

1. 在 VS Code 中打开 `01_mlp_lab/00_一键体验.ipynb`；
2. 选择 `dl2026` Python 内核；
3. 点击 `Run All`；
4. 先观察结果表和图片，再继续打开下一份 Notebook。

核心实验运行时不需要下载数据或模型，配置完成后可以离线使用。

## 🗺️ 接下来还会有什么？

| 实验室 | 状态 | 主要内容 | 计划展示 |
|---|---|---|---|
| `01_mlp_lab` | ✅ 已完成 | MLP、激活函数、优化器和二维分类 | 决策边界、Loss、Accuracy、梯度 |
| `1.1` MLP 房价预测 | ✅ 已完成 | 多特征回归、数据划分、参数比较 | 房源表、预测阶段、误差图、自定义房源预测 |
| `02_cnn_lab` | ✅ 已完成 | 卷积、池化、特征提取和图像分类 | 卷积核、特征图、混淆矩阵、错误样本 |
| `03_rnn_lab` | ✅ 已完成 | RNN、LSTM、GRU 和长期依赖 | 秘密时间线、隐藏状态、梯度、长度对比 |
| `04_transformer_lab` | ✅ 已完成 | Token、位置编码、Self-Attention 和小型 Encoder | Q/K/V 手算、Attention 图、组件对比、四小时挑战 |

后续实验室会继续采用“基础六步 + 综合项目”的路线：先快速看到结果，再理解数据、
训练规则和训练过程，随后完成组件对比、自主修改与证据报告。

## 📚 第一次配置？按设备选择

### Windows 自带电脑

1. [Windows 自带电脑：Conda 与 PyTorch 安装指南](./5.Windows自带电脑Conda与PyTorch安装.md)
2. [只读 Windows 环境预检脚本](./check_windows_environment.ps1)

CPU 是所有 Windows 电脑的保底路线；只有 `nvidia-smi` 正常的 NVIDIA 电脑才选择 CUDA wheel。AMD / Intel 显卡在本课程的 Windows 标准路线中使用 CPU。

### Linux 课程服务器

1. [VS Code Remote-SSH 连接 Linux 服务器与失败排查](./1.1%28补充材料%29VSCode%20Remote-SSH连接Linux服务器与失败排查.md)
2. [GitHub 项目在 Linux 服务器上的拉取与更新](./2.1%28补充材料%29GitHub项目在Linux服务器上的拉取与更新.md)
3. [Linux 服务器 Conda 与 PyTorch 安装](./3.Linux服务器Conda与PyTorch安装.md)
4. [Linux 与 Conda 常用命令](./4.Linux与Conda常用命令.md)

公开文档中的服务器 IP、端口和登录信息使用占位符，实际连接信息以课堂说明为准。Windows 安装命令是带核对日期的课程快照；PyTorch 版本发生变化时，以指南链接的官方选择器为准。

## 📁 为什么项目里有两份 README？

- 根目录的 [README.md](./README.md)：GitHub 仓库首页，介绍整个课程项目、学习路线和建设计划；
- [01_mlp_lab/README.md](./01_mlp_lab/README.md)：MLP 子项目说明，介绍七份核心 Notebook、代码结构和具体运行方法；
- [02_cnn_lab/README.md](./02_cnn_lab/README.md)、[03_rnn_lab/README.md](./03_rnn_lab/README.md) 和 [04_transformer_lab/README.md](./04_transformer_lab/README.md)：分别提供 CNN、RNN 与 Transformer 实验室的具体入口。

每个已完成的实验室都有自己的 README。根目录 README 负责“带你找到方向”，子目录 README 负责“带你完成这个实验”。

## 💡 一个小建议

第一次运行时，不必急着理解 `engine.py` 或 `plots.py`。先完成六份 Notebook；准备修改模型结构时，再阅读短小的 `01_mlp_lab/mlp_lab/model.py`。

先跑起来，先看见变化，再去理解代码——深度学习会清楚很多。🚀
