# 小型 Transformer 注意力寻宝实验室

这个实验室不用大语言模型，也不下载语料。每条样本只有红、绿、蓝三种颜色 token，末尾追加一个 `QUERY`；模型需要找回第 0 位的颜色。

每条序列中三种颜色的总数完全相同，因此不使用位置编码的模型不能靠计数猜答案。数据由本地代码生成，训练、验证、测试集标签平衡且无精确 token 序列重叠。

## 推荐顺序

| 顺序 | Notebook | 主要任务 | 主要证据 |
|---:|---|---|---|
| 0 | `00_一键体验.ipynb` | 先完成一次注意力寻宝 | 彩色 token、训练曲线、混淆矩阵、Attention |
| 1 | `01_Token与三维张量.ipynb` | 对齐 token id、Embedding 与 shape | `[B,T]` 到 `[B,T,D]` |
| 2 | `02_位置编码与QKV.ipynb` | 手算一次 Scaled Dot-product Attention | 位置编码热力图、权重柱状图 |
| 3 | `03_训练第一个Transformer.ipynb` | 拆解完整训练过程 | Loss、Accuracy、梯度、预测概率 |
| 4 | `04_Transformer组件对比.ipynb` | 比较位置编码与 Head 数 | 受控实验表、预算图 |
| 5 | `05_自主修改练习.ipynb` | 修改一处激活映射 | 基线、唯一修改、烟雾测试 |
| 6 | `06_注意力寻宝项目.ipynb` | 四小时综合挑战 | 四枚徽章、三组对照、代码与证据报告 |

## 最短启动

```bash
conda activate dl2026
cd "$HOME/2026DeeplearningDemo/04_transformer_lab"
python test_transformer_smoke.py
python demo.py --experiment baseline --device cpu
```

然后打开 `00_一键体验.ipynb`，选择 `Python (dl2026)` 并点击 `Run All`。

## 共享服务器安全门

- 默认使用 CPU、`torch.set_num_threads(1)`、`DataLoader(num_workers=0)`；
- 共享入口拒绝 `device="auto"` 和无编号的 `device="cuda"`；只有教师明确分配后才使用 `cuda:x`；
- `content_length≤24`、`train_size≤3000`、`epochs≤30`、`batch_size≤128`；
- `d_model≤64`、`num_heads≤4`、`num_layers≤2`、`dim_feedforward≤128`；
- 参数更新次数≤1500、参数量≤150000；
- `train_size × epochs × num_layers × input_length²≤12000000`。

这些上限在创建数据和分配模型前检查。四小时工作量来自观察、手算、对照、代码修改和证据报告，不来自延长训练。

本机课程环境实测默认基线约 1 秒、测试 Accuracy 约 99.7%；这只是当前机器和固定 seed 的校准值，不是服务器性能承诺，也不是学生必须复现的成绩。

## 第一版明确不做

不引入 `transformers`、`torchtext`、Tokenizer、预训练模型、在线数据、Decoder、Causal Mask、文本生成、变长 Padding 或大词表。Attention 图只说明当前前向计算中的权重分配，不能当作模型因果解释。

## 资源释放

每份 Notebook 最后一格都会清空变量、关闭图片、释放未使用的 CUDA 缓存并结束内核。再次实验时重新选择课程内核即可。
