# CNN 手写数字与特征图实验室

本实验室使用 scikit-learn 自带的 `load_digits`，无需联网下载数据，也不依赖 `torchvision`。
第一版从 `8×8` 灰度数字开始，但模型、数据对象和绘图均保留动态图片尺寸，可在后续接入课前准备好的 `28×28` 手写数字或手写字符。

## 推荐顺序

| 顺序 | Notebook | 主要任务 | 主要图片 |
|---:|---|---|---|
| 0 | `00_一键体验.ipynb` | 先完成一次数字分类 | 训练诊断、混淆矩阵、错误样本、Feature maps |
| 1 | `01_图片数据与四维张量.ipynb` | 对齐图片、像素和 `[B,C,H,W]` | 数字画廊、像素矩阵、shape 流程 |
| 2 | `02_卷积核与池化.ipynb` | 理解局部卷积、Padding 和 Pooling | 人工卷积核、输出响应、池化对比 |
| 3 | `03_训练第一个CNN.ipynb` | 训练两层 CNN 并观察中间层 | Loss、Accuracy、混淆矩阵、两层 Feature maps |
| 4 | `04_CNN组件对比实验.ipynb` | 单独比较三种 Pooling | 准确率、平移测试、参数量和曲线 |
| 5 | `05_自主修改练习.ipynb` | 只修改一个结构参数 | 基线与修改后诊断图 |
| 6 | `06_CNN识别侦探项目.ipynb` | 四小时综合取证、最小代码修改和预算挑战 | 四徽章、混淆数字、Feature maps |

## 四小时综合项目

`06_CNN识别侦探项目.ipynb` 把 240 分钟分配给像素观察、卷积与池化解释、三组
单变量实验、阅读并修改 `model.py`、预算内调参和证据报告。全项目正式训练最多
8 次；四小时来自分析、编程和表达，不来自长时间占用服务器。

四枚徽章分别检查原图 Accuracy、右移一像素后的 Accuracy、参数量和更新次数。
评分不使用运行时间。训练入口会拦截 `device="auto"`，并限制
`epochs≤40`、`batch_size≤256`、通道数≤64、`kernel_size≤7`、单次更新≤4000、
模型参数量≤750000。默认仍为 CPU 单线程，数据离线加载，DataLoader 不创建子进程。

模型现在把激活函数集中在 `cnn_lab/model.py` 的 `_activation()`，学生增加一种
PyTorch 已有激活函数时只需修改一个映射，不需要复制模型或训练循环。

## 最短启动

```bash
conda activate dl2026
cd "$HOME/2026DeeplearningDemo/02_cnn_lab"
python test_cnn_smoke.py
```

然后在 VS Code 中打开 `00_一键体验.ipynb`，选择 `Python (dl2026)` 并点击 `Run All`。

共享服务器默认使用 CPU。只有获得课堂分配后，才把 Notebook 中的 `DEVICE = "cpu"`
改成 `DEVICE = "cuda:x"`，并用指定的 GPU 编号替换 `x`。编号从 0 开始。

## 28×28 扩展方式

后续确定 28×28 数据后，只需新增一个数据加载函数，返回与 `load_digits_data()` 相同的
`ImageDatasetBundle`：图片必须为 `[N,C,H,W]`，同时提供标签、类别名称和像素范围。
`SmallCNN(image_size=28)` 已通过前向测试；训练、设备选择和主要绘图接口不需要复制。

正式课堂不临时下载数据。28×28 数据应提前放入仓库或课程材料，并提供文件校验。

## 资源释放

每份 Notebook 最后都有“清理并结束内核”单元格。运行它会清空变量、释放未使用的
CUDA 缓存，并正常结束当前 Python 进程。VS Code 随后显示内核停止或要求重新选择内核属于正常现象。
