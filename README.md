# 🧠 深度学习课堂 Demo

> 不从一大段代码开始，而是先点一下、看到结果，再一步步理解模型为什么会这样表现。

欢迎来到这个“讲一点、跑一下、改一处、看一张图”的深度学习实验项目！🎉

这里的目标不是背代码，也不是只追求更高的 Accuracy，而是通过可以直接运行的 Notebook，真正看懂数据、模型、训练过程和实验对比。

## 🎯 这个项目是做什么的？

每个实验室都会把一个深度学习主题拆成六个小步骤：

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

六份 Notebook 都保存了一次成功运行的图形输出。即使暂时没有配置好运行环境，也可以先在 GitHub 上打开它们预览实验结果。👀

更详细的 MLP 使用说明见 [01_mlp_lab/README.md](./01_mlp_lab/README.md)。

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

更详细的 CNN 使用说明见 [02_cnn_lab/README.md](./02_cnn_lab/README.md)。

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

## 🚀 最快开始方式

### 第一步：获取项目

```bash
git clone https://github.com/AYHQChang/2026DeeplearningDemo.git
cd 2026DeeplearningDemo
```

### 第二步：进入课程环境并检查

```bash
conda activate dl2026
python check_environment.py
python 01_mlp_lab/test_mlp_smoke.py
python 02_cnn_lab/test_cnn_smoke.py
```

看到以下内容，说明基本环境和 MLP 实验可以运行：

```text
环境自检通过，可以离线运行课堂 Demo。
MLP smoke test passed.
CNN smoke test passed.
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
| `02_cnn_lab` | ✅ 已完成 | 卷积、池化、特征提取和图像分类 | 卷积核、特征图、混淆矩阵、错误样本 |
| `03_rnn_lab` | 🧭 待建设 | RNN、LSTM、GRU 和长期依赖 | 序列预测、梯度、长度对比、训练时间 |

后续实验室会继续采用相同的六步路线：先快速看到结果，再理解数据、训练规则和训练过程，最后完成组件对比与自主修改。

## 📚 第一次使用服务器？按这个顺序阅读

1. [VS Code Remote-SSH 连接 Linux 服务器与失败排查](./1.1%28补充材料%29VSCode%20Remote-SSH连接Linux服务器与失败排查.md)
2. [GitHub 项目在 Linux 服务器上的拉取与更新](./2.1%28补充材料%29GitHub项目在Linux服务器上的拉取与更新.md)
3. [Linux 服务器 Conda 与 PyTorch 安装](./3.Linux服务器Conda与PyTorch安装.md)
4. [Linux 与 Conda 常用命令](./4.Linux与Conda常用命令.md)

公开文档中的服务器 IP、端口和登录信息使用占位符，实际连接信息以课堂说明为准。

## 📁 为什么项目里有两份 README？

- 根目录的 [README.md](./README.md)：GitHub 仓库首页，介绍整个课程项目、学习路线和建设计划；
- [01_mlp_lab/README.md](./01_mlp_lab/README.md)：MLP 子项目说明，介绍六份 Notebook、代码结构和具体运行方法。

每个已完成的实验室都有自己的 README；以后新增 `03_rnn_lab` 时也会沿用这一结构。根目录 README 负责“带你找到方向”，子目录 README 负责“带你完成这个实验”。

## 💡 一个小建议

第一次运行时，不必急着理解 `engine.py` 或 `plots.py`。先完成六份 Notebook；准备修改模型结构时，再阅读短小的 `01_mlp_lab/mlp_lab/model.py`。

先跑起来，先看见变化，再去理解代码——深度学习会清楚很多。🚀
