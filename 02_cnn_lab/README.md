# CNN 手写数字与服饰分类实验室

本实验室先用 scikit-learn 内置的 `load_digits` 完成 `8×8` 灰度数字分类，再用项目提供的 `28×28` Fashion-MNIST 图片练习手动替换数据集。两个任务使用相同的网络组合和训练入口，不需要安装 `torchvision`。

## 学习顺序与二选一任务

| 阶段 | Notebook | 使用者要完成的事 |
|---|---|---|
| 基础 | `00_一键体验.ipynb` | 先跑通一次数字分类，找到训练曲线、混淆矩阵和 Feature maps |
| 基础 | `01_图片数据与四维张量.ipynb` | 对齐像素、标签与 `[B,C,H,W]` |
| 基础 | `02_卷积核与池化.ipynb` | 理解卷积、Padding、Max/Avg Pooling |
| 基础 | `03_训练第一个CNN.ipynb` | 理解两层 CNN 的训练与中间响应 |
| 基础 | `04_CNN组件对比实验.ipynb` | 固定其余条件，比较三种池化方式 |
| **路线 A：数字** | `05_自主修改练习.ipynb` | 自己组合数字分类网络，完成基线、结构变化、训练变化、二者组合共 4 组实验 |
| **路线 B：Fashion** | `07_手动替换图片数据集.ipynb` | 手动接入服饰图片，完成至少 3 种网络结构的验证集对比，再选定一种做最终测试，并练习从三类改为两类 |
| 进阶选做 | `06_CNN识别侦探项目.ipynb` | 阅读源码、增加激活函数、预算内探索并解释证据 |

建议先完成 **00–04** 的基础实验，再从 **05 数字路线或 07 Fashion 路线中任选一条** 完成课程实验，之后可选择 06 进阶练习。两条路线二选一，只需提交所选路线的一份实验报告；未选路线无需完成或写入报告。这里的“手写字”具体指 0–9 手写数字；服饰入门包包含包、运动鞋、裤子三个类别。

## 最短启动

在服务器终端执行：

```bash
conda activate dl2026
cd "$HOME/courses/Deeplearning2026/02_cnn_lab"
python test_cnn_smoke.py
python test_cnn_composition.py
```

在 VS Code 的“文件 → 打开文件夹”中直接打开 `02_cnn_lab`，再打开 Notebook，选择 `Python (dl2026)` 内核。首次运行从第一个单元格开始顺序执行。00–05 可用 `Run All`；07 学生版含必须补全的 TODO，初始直接 `Run All` 会在明确提示处停止，需逐项完成后再继续。项目根目录也能运行；不要只打开一个游离的 Notebook 文件或其数据子文件夹。若提示 `No module named cnn_lab`，先检查打开的文件夹和当前内核，而不是执行 `pip install cnn_lab`。

共享服务器默认 `DEVICE = "cpu"`。获得课堂分配后，才使用指定编号的 `cuda:x`；编号从 0 开始。

## 自己组合 CNN：要改哪里

05 和 07 都把操作集中在三个位置：**数据设置 → 网络结构 → 训练设置**。网络通过 `ConvBlock` 列表组合，每个块按“卷积 → 可选 BatchNorm → 激活 → 可选池化 → 可选 Dropout”运行。

```python
BLOCKS = (
    ConvBlock(out_channels=4, kernel_size=3, activation="relu", pooling="max"),
    ConvBlock(out_channels=8, kernel_size=3, activation="relu", pooling="none"),
)
```

每一行是一个卷积块；增加或删除一行就改变网络深度。允许 1–3 个块，分类层根据图片尺寸和类别数量自动建立。

| 参数 | 可选值或范围 | 建议观察什么 |
|---|---|---|
| `out_channels` | 正整数，最大 64 | 参数量、每层 Feature maps、验证准确率 |
| `kernel_size` | `1`、`3`、`5`、`7` | 卷积覆盖范围；Padding 保持卷积前后的空间尺寸 |
| `activation` | `"relu"`、`"tanh"`、`"leaky_relu"` | Loss 与验证曲线；可给每个块选择不同函数 |
| `pooling` | `"max"`、`"avg"`、`"none"` | 空间尺寸和参数量；池化会把边长减半并向下取整 |
| `batch_norm` | `True` 或 `False` | 同一设置下的训练变化；进阶选项 |
| 块内 `dropout` | `0 ≤ 值 < 1`，初学可试 `0.1`、`0.2` | 训练与验证准确率的差距 |
| `ExperimentConfig.dropout` | `0 ≤ 值 < 1` | 分类头的 Dropout，与块内 Dropout 分开设置 |

块内 Dropout 使用 `Dropout2d`，训练时随机屏蔽部分特征通道；分类头 Dropout 作用于展平后的特征。验证与测试时，训练入口会关闭 Dropout，并让 BatchNorm 使用累计统计。

结构配置写入 `ExperimentConfig(blocks=BLOCKS, ...)`。填写 `blocks` 后，旧的 `channels/kernel_size/pooling/activation` 简写不再决定卷积结构；00–04 的原有两层简写仍可使用。`learning_rate`、`batch_size`、`epochs`、`seed`、`device` 仍在训练配置里修改。

先查看 Notebook 输出的模型结构、逐层形状和参数量，再开始训练。不要把“多堆一层”当成必然提升；例如池化太多会把图片压得过小。填写偶数卷积核、超过 3 个块、非法参数或会使图片尺寸变为 0 的池化时，程序会给出配置错误提示。

## 比较模型与最终测试

05 使用数字训练/验证/测试划分，07 使用图片目录中的三个划分。探索阶段统一调用：

```python
from datetime import datetime

# 每一批正式对比开始时生成一次路径；这一批内使用同一个 record_path。
record_path = f"outputs/my_cnn_trials_{datetime.now():%Y%m%d_%H%M%S}.csv"
result = train_experiment(config, data=data, evaluate_test=False)
```

每次修改结构或训练参数，都保留实验名称、配置、参数量和验证成绩。05 的训练因素可选择 `learning_rate`、`batch_size` 或分类头 `dropout`；四组只改变对应因素，共同固定数据划分、`epochs`、`seed` 和 `device`。若改变 `batch_size`，报告中需说明每轮及总更新次数也会变化。通过验证集和模型大小选定一个方案，然后运行一次：

```python
evaluate_final(selected)
save_experiment_records(results, record_path)
```

CSV 文件名应包含训练批次的时间戳，例如 `outputs/digits_trials_20261007_143000.csv`，重新训练一批时生成新文件，便于保留之前的记录。完成最终测试后，用同一批 `results` 再保存到本轮 CSV，使选定模型的测试成绩回写当前记录；未选中的模型仍保留空测试成绩。

最终测试的作用是报告选定模型在保留数据上的表现。测试后不要继续根据测试成绩调参。探索阶段的混淆矩阵、错误图片和 Feature maps 使用验证集；最终测试后可查看所选模型的测试诊断图。

若想完全自己写网络，可在 Notebook 定义 PyTorch `nn.Module`，通过 `train_experiment(config, data=data, model=my_model, evaluate_test=False)` 训练。**先调用 `seed_everything(config.seed)`，再创建自定义模型**，才能固定模型初始化。模型应接收 `[B,1,H,W]` 并输出 `[B,类别数]` 的原始分类分数；不要在最后加 Softmax。自定义模型若提供 `feature_maps(x)`，也可用现有函数查看中间特征。

## 手动替换服饰或自己的图片

第 07 份 Notebook 是学生版数据迁移练习。教师提供图片包、单张图片示范、目标形状和分层提示；学生独立填写数据路径、类别/标签映射、批量读取中的归一化、通道与标签，以及 `ImageDatasetBundle` 组装。未完成的 TODO 会给出具体提示并停止，不自动代填答案或切回数字数据。

```text
my_dataset/
├── train/
│   ├── bag/       图片.png 或 图片.jpg
│   ├── sneaker/
│   └── trousers/
├── val/          与 train 相同的类别文件夹
└── test/         与 train 相同的类别文件夹
```

修改 `DATA_DIR`、`CLASS_NAMES` 与需要时的 `IMAGE_SIZE`，重新运行数据加载与检查单元格，再运行模型定义和训练。三个划分必须使用相同的类别名称；标签由 `CLASS_NAMES` 的顺序决定，每个划分的每类都应有图片。不要把同一张图片复制到不同划分，也不要用验证/测试图片补训练集。图片统一转成单通道正方形，彩色图会转为灰度。

07 的准备单元格会将 `data/fashion_small.zip` 解压到 `data/fashion_small/`，随后由学生自行选择数据路径并完成接入。课程包选自 [Zalando Research Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)（MIT 许可，见 `data/FASHION_MNIST_LICENSE.txt`）：`bag`、`sneaker`、`trousers` 三类，每类训练 300、验证 100、测试 100 张，共 1500 张 28×28 灰度 PNG。训练/验证图来自官方训练集，测试图来自官方测试集；固定随机种子为 42。压缩包内的 `MANIFEST.json` 保存源文件 SHA256 与原始索引。教师可用 `tools/prepare_fashion_subset.py` 重建数据包，正常实验无需联网。

## 修改后运行哪些单元格

| 修改内容 | 操作顺序 |
|---|---|
| `BLOCKS` 或训练参数 | 重跑配置/模型预览 → 对应训练与记录 → 验证对比；保持已有实验记录 |
| 数据目录、类别、图片尺寸 | 重跑数据加载与检查 → 模型配置/预览 → 训练与记录；为新数据开始一组新记录 |
| `cnn_lab/model.py` 等 `.py` 源码 | 保存文件 → 重启 Notebook 内核 → 重跑导入、数据、配置及需要的训练单元格 |
| 只修改报告或分析文字 | 重新查看已有图表和实验记录即可 |

06 中新增激活函数后，先在**终端**运行 `python test_cnn_smoke.py`，再按上表重启 Notebook 内核。已导入模块会留在内存里，仅重跑训练单元格不一定载入新源码。重启后变量和实验列表会清空，必要记录应先保存为 CSV。

## 报告与提交

从 **05 数字路线与 07 Fashion 路线中二选一**，只提交 **所选路线的一份 Word 实验报告**，使用教师提供的 `02_CNN模型组合探索实验报告模板.docx`。未选路线无需完成或写入报告。

- 选择 **05 数字路线**：报告记录基线、结构变化、训练变化、二者组合共 4 组实验。
- 选择 **07 Fashion 路线**：报告记录手动数据接入过程、至少 3 种网络结构的验证集对比，并展示数据形状、类别映射与分类输出检查。完成三类接入后，可继续探索两类或自己的图片。

所选路线都应保留配置表、验证对比、选定模型的最终测试结果，以及能支撑解释的曲线、混淆矩阵、错误样本或 Feature maps。

Notebook 和自动生成的 CSV 保存在自己的项目目录，作为复查依据；默认无需额外提交，教师另有要求时再附上。源码挑战只在实际修改了代码时保留相应源码文件。评分规则由教师课堂公布。项目保留多组准确率、配置、曲线和数据接入证据，供比较与复查。报告模板可从 [02_CNN模型组合探索实验报告模板.docx](../docs/02_CNN模型组合探索实验报告模板.docx) 获取。

## 多组结果与自由迁移

05 可记录四组结构/训练因素实验，07 可以在 `MODEL_PLANS` 中增加或修改多种网络；每次训练保留配置、参数量、训练/验证准确率和独立 CSV。不同批次使用新文件名，避免覆盖旧记录。先根据验证集选择模型，再计算其最终测试准确率；不要依据测试成绩反复挑模型。

课程默认数据、seed 和轮数是起步配置，不是自动评分限制。可以在共享上限内改变网络层数、通道、卷积核、激活、池化、BatchNorm、Dropout、学习率、Batch size 和 Epoch。进行同一组组件对比时固定数据划分、seed、轮数和其余训练条件；改变多个因素时明确记录，避免声称单因素结论。

07 完成初次 Fashion 接入后，还可自己修改类别、图片尺寸和数据目录，重新完成数据检查与所有训练，观察两类或自定义图片任务。相同数据接口继续支持不同 CNN 组合；各任务难度和类别数不同，不能直接以准确率高低判断模型更强。自己整理图片时保留独立 train/val/test，避免重复图片跨划分。

`validate_image_dataset(data)` 只检查学生自己组装的张量、类别、形状、像素范围和各划分每类样本数，不加载图片或代填接入答案，也不锁定数据集或自动计分。

## 预算与资源释放

CPU 单线程，DataLoader 不创建子进程。单次实验限制 `epochs≤40`、`batch_size≤256`、每层通道数≤64、卷积核≤7、更新次数≤4000、模型参数量≤750000。建议先用少量轮数确认流程，再执行正式对比；四小时来自观察、编程与解释。06 的正式训练最多 8 次。

每份 Notebook 最后有“清理并结束内核”单元格，会清空变量、释放未使用的 CUDA 缓存并结束 Python 进程。先保存 Notebook、CSV 和报告图片，再运行它。VS Code 随后显示内核停止或要求重新选择内核属于正常现象。
