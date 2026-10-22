# 中文评论情感分类 三级实验

[学校报告模板](../../docs/03_RNN模型组合探索实验报告模板.docx)

统一作业入口：[03 调参 完善 自行建模](03_自主修改与挑战.ipynb)。[01编码](01_文字如何变成序列.ipynb)可直接运行；00与02是教师参考，默认关闭重算。历史参考保留在[reference](reference/README.md)，不能作为个人成绩。

使用现有课程环境和教师提供的本地数据，默认CPU单线程。修改模型或配置后重启内核从头运行；每次写入新的outputs/text/three_levels/时间_uuid目录。

## 当前作业与评分规则（2026-10-22）

学生入口为文本03，保留“调参—完善—自行建模”三级。正式成绩 = 100 × 最终测试Accuracy（0到1）×等级系数：一级0.8、二级0.9、三级1.0。Loss、Macro F1、参数量和耗时用于解释，不另加权。程序仅给待教师验收的折算分；等级按实际代码与证据认定，系数跟随所选模型，不能借用另一等级的系数。

| 等级 | 完成要求 |
|---|---|
| 一级 调参 | 普通RNN基线、两个学习率候选、至少一次依据曲线提出的后续单参数实验，共至少4组；记录运行前预测和运行后结论 |
| 二级 完善 | 补全LSTM三处TODO，先通过形状/PAD/有效读出语义/梯度连接检查，再与基线RNN比较并分析错例 |
| 三级 自行建模 | 自己实现GRU分类器；同一GRU的A/B仅改变读出、H、E或层数之一，保持训练条件一致；两组均检查、训练并解释设计依据 |

默认只运行一级的三组学习率粗筛；后续调参需学生填写，二/三级开关默认关闭，模型TODO保留。三级独立于二级，允许nn.GRU，不要求手写门控。仅换LSTM为GRU、只提交一个GRU或只有漂亮曲线均不足以完成三级。效果下降也可完成实验，但必须如实解释。

调参先粗筛学习率，再依据曲线、梯度和截断诊断决定下一项；不要同时改多个因素。支持Adam（默认）、SGD+Momentum（默认0.9）和选做AdamW；同时提供weight_decay、gradient_clip和多层循环dropout。单层层间dropout不生效。更换优化器后需检验适合它的学习率，不从同一学习率的单次结果推断普遍优劣。改batch时更新次数随之变化，CSV明确记录。

学生组明确使用best_val_accuracy：选验证Accuracy最高轮；同分选Loss更低轮，再同分保留早轮。恢复该轮模型后导出验证指标、预测和最终测试。旧教师参考默认selection=last以维持历史流程，两者不能混称。history保留所有训练轮，selected_epoch记录所选轮次。

正式提交前先在验证集选定一组，固定配置、seed=42、保存轮次和选择理由，写入selection_frozen.json后才评价测试；探索时关闭测试，正式提交必须完成最终测试。不要重建实验来按测试成绩反复选型。其他种子仅作为预先声明的验证稳定性复核，不挑最好种子报成绩。

提交学校格式Word主报告，并附个人Notebook与all_experiments.csv。运行目录保留模型、预处理、history、接口检查、预测、设计对照图和最终选择记录供核验。Word中的Accuracy/F1填百分数；代码里的Accuracy是0到1。原封面与学校填写字段保留。

课堂顺序：文本00看教师参考→文本01编码→文本03一级；进入二级前完成原理04的返回值形状与最小分类器过渡练习。当前语义检查用不同时间位置的人工数值验证读出，并用小批次反传检查计算图；不更新权重、不替代教师验收，不提供学生模型答案。

## 数据：固定版本、真实标签与原有划分

本地课程包版本为 **chnsenticorp-paddlehub-v2.0.0-unique-course-v1**，使用 [PaddleNLP v2.0.0 的 ChnSentiCorp 加载器](https://github.com/PaddlePaddle/PaddleNLP/blob/v2.0.0/paddlenlp/datasets/chnsenticorp.py)指向的[有标签 tar.gz 数据](https://bj.bcebos.com/paddlehub-dataset/chnsenticorp.tar.gz)。没有混入当前另一个同名 zip 版本，其 test 标签条件也不同。PyTorch 负责模型训练，不安装 Paddle。

标签来自原文件：`0=负面`、`1=正面`。原始文件现场核对为 train=9600、dev=1200、test=1200。先对所有原划分文本采用统一的 NFKC/空白规范化检查完全相同文本：共2039个重复文本组，其中694组跨划分、36组存在标签冲突。**任意重复文本组的全部出现均排除**，不只保留一份，不迁移划分，不改标签，也不根据模型成绩挑样本。排除后可用 train=6284、dev=790、test=793。

随后在各原划分内部，以固定 seed=42 按原标签分层抽样，不重新切分，不把 dev/test 混进 train：

| 课程文件 | 原划分 | 实际数量 | 负面/正面 |
|---|---|---:|---:|
| `data/train.tsv` | train | 2000 | 1000 / 1000 |
| `data/val.tsv` | dev | 500 | 250 / 250 |
| `data/test.tsv` | test | 500 | 250 / 250 |

样本 ID 保存原划分与源行号；dev 只在课程接口中改名 val。课程三划分的规范化完全相同文本重叠为0，尚未排查语义近重复。排除重复使课程子集分布与原始数据不同，结果不能直接当作完整 ChnSentiCorp 基准。教师编写的小例句只用于编码讲解和额外输入，未混进正式训练、验证或测试集。

[manifest.json](data/manifest.json)保存源文件哈希、实际数量、抽样算法、排除规则和课程 TSV 的 SHA256。固定原始压缩包的 SHA256 是 `7251a6c7bedf7116b4727c809a694405187e1e9cbdb770f509504b16ca6d9c53`，MD5 是 `fbb3217aeac76a2840d2d5cd19688b07`。复现使用[prepare_data.py](prepare_data.py)，只接受本地源包并核验固定哈希，使用标准库且不自动下载。示例从仓库根运行：

```bash
python 03_rnn_lab/06_text_classification/prepare_data.py 本地源包路径/chnsenticorp.tar.gz
```

本地已准备 [data/chnsenticorp_course_v1_offline.zip](data/chnsenticorp_course_v1_offline.zip)，将其中 train.tsv、val.tsv、test.tsv 和 manifest.json 解压到本目录的 `data/` 即可离线运行。学生应由教师按数据允许的范围提供课程包；代码本身不从公网取数据。若只克隆公开代码而没有 TSV，请先按教师提供的合法课程数据路径准备，不能拿人工例句代替正式训练集。

**使用和再分发边界：**固定旧 tar 包内没有数据许可文件，PaddleNLP 源代码的 Apache 许可不能作为评论数据的许可。另一个当前 LUGE zip 包的 Participant Agreement 涉及非商业科研/教学，但不能直接套用为这个固定旧包的明确公共再分发授权。本轮仅准备本地非商业教学材料；未确认公开上传、公开再分发或面向学生分发的授权。评论 TSV、源压缩包和课程 zip 已 gitignored；公开发布或分发前须按数据条件核实，不能因代码仓库可公开就默认评论可公开。清单和制作脚本可保留用于复现。

## 按字编码与有效长度

统一清理规则 `nfkc-whitespace-v1`：NFKC 统一兼容/全半角字符（例如全角叹号变半角），连续空白压成一个空格并去掉首尾空白。保留字符顺序、否定、转折与标点，不盲目删停用词。只从课程 train 构建词表，固定 `<PAD>=0`、`<UNK>=1`；默认其余字符按频数降序、同频按字符排序。验证、测试及用户输入复用同一词表，不临时扩词。

```text
原文 → 字符 → ID [B,T]
     → 随机初始化且参与训练的 Embedding [B,T,E]
     → 单向循环层顶层输出 [B,T,H]
     → 有效 last 或 masked hidden mean [B,H]
     → Linear [B,K] logits → 观察时 softmax 概率和类别
```

K 从数据配置推导，本包为2。批次返回 int64 的 `token_ids:[B,T]`、`lengths:[B]`、`labels:[B]`，另保留原文/ID。右侧补 PAD；T 是该批最长有效长度，lengths 是截断后的真实长度。空/清理后为空文本用一个 UNK 哨兵表示，length=1、empty=True；全 UNK 保留未知字计数；超长文本按 max_length 截断并标记 truncated。使用者应检查这些字段，空或全未知输入的数值输出不能视为可靠情感判断。

`padding_idx=0` 只处理查表中的 PAD 行，不会让循环层自动忽略 PAD。公共模型将右侧补齐后的批次送入单向循环层；last 按 lengths−1 索引每条最后有效字符，mean 仅沿有效时间位置平均。右侧 PAD 不影响前面有效位置，但补齐后的最终隐藏状态可能已处理 PAD，不能无条件拿它代替有效句尾。第一版不支持双向，不能把这里的读出直接推广到双向网络。平均 Embedding 基线排除 PAD 但不利用顺序；它与平均 RNN 隐藏状态是两种不同模型。

`save_text_run` 每次创建新目录并保存词表/编码配置/类别映射、config、模型 state_dict、history.csv、experiments.csv、metrics.json。`load_text_preprocessing` 可读取已保存的同版本配置；用户新输入使用 `predict_text(result, text)`，输出原文、编码长度、截断、未知字、预测类别和各类模型概率。概率未做概率校准，不称作可信度。


## 2026-10-22 本轮验证

八个RNN测试入口通过，包括错误取首状态、断梯度、状态保持、最佳轮次恢复、优化器、等级折算和单因素对照。10本当前Notebook通过schema/语法检查；学生文本03默认执行成功，仅运行一级三组粗筛。另在本地忽略目录的临时副本补入教师实现，完成后续调参、LSTM与GRU last/mean两组训练及导出。学生源本的3+2处模型TODO逐字保持，测试集未用于本轮正式数据选型或评价；评分链路仅用人工测试夹具核验。

学校报告为6页，学校目录与仓库副本逐字节一致；通过隐藏Word只读导出及打包Poppler逐页检查，封面与原模板像素一致。Windows本机CPU单线程验证；本轮未验证Linux服务器、GPU或并发。临时答案、执行副本、QA图和数据包不随Git发布。新版学生代码输出清空，两个明确标注的历史图保留，不把旧成绩当新版结果。

## 2026-10-19 三级实验验证

- 学生03默认从仓库根执行成功：仅完成普通RNN基线和单参数学习率对照，6轮、E=H=32、max_length=100；验证Accuracy为0.648与0.594，Loss为0.70373与0.69667。调参未提高准确率，按真实结果保留，不设置达标线。
- 二级原始构造TODO明确停止；分步临时补齐后，前向TODO和有效句尾TODO分别明确停止。三级原始构造同样明确停止。学生源本的3+2处缺口保留。
- 仅在本地忽略目录的执行副本中填教师临时实现，验证LSTM/GRU共同完成，以及关闭LSTM、单独运行GRU（含masked mean设计）的完整训练、预测、CSV导出。上述临时实现和成绩不回填学生文件，也不作为学生成绩。
- 00/02的参考开关关闭时成功执行并确认没有训练调用，01图解执行成功；源本19幅已有文本图输出保持，四本原文件归档与修改前逐字节一致。当前03含11幅新增运行/图解输出和2幅历史图。
- 原有6个测试入口及新增test_rnn_course.py均通过；检查缺失学生模型不会回退、PAD/混批不变性、真实模型类型和未测测试列留空。
- Word模板4页，使用独立隐藏Microsoft Word只读导出工作副本，再由bundle Python/Poppler转PNG逐页核对；未使用系统LibreOffice。未测试跨平台Word排版。
- 本次使用Windows本机Ayitedu环境、CPU单线程；未验证Linux服务器、GPU或多人并发。正式测试集未评价。历史数据、输出和用户无关文件保持；未提交或推送Git。

本地验证记录在实验室`outputs/three_level_20261019/`：notebook_validation.json、test_report.json、delivery_verification.json及report_work/report_qa.json。新增实际运行记录位于实验室`outputs/text/three_levels/`。

## 2026-10-17 参数与CPU校准 历史参考

可改参数集中在 Notebook 的 `TextExperimentConfig`：cell_type、embedding_dim、hidden_size、num_layers、readout、max_length、epochs、batch_size、learning_rate、seed、dropout、gradient_clip 和 device。当前固定单向、默认零初态，普通 RNN 为 tanh，交叉熵训练直接接 logits，优化器 Adam。循环 dropout 只作用于循环层之间，单层时不会生效；不同循环单元在相同 H 时参数量不同。

当前默认起点 E=32、H=32、1层、batch=64、max_length=100、6轮、学习率0.003、seed=42、CPU单线程。2026-10-17 在本机 **Python 3.11.14、PyTorch 2.11.0+cu128** 上独立校准了一组 GRU：6轮训练及每轮验证共 **2.561秒**，完整 `run_text_experiment` 为3.525秒，词表3081、实际参数104994。训练Loss由0.68869降至0.33021，验证Loss=0.42182、Accuracy=0.806、宏平均F1=0.80596，测试未评价。这些是当前小包、配置与机器的一次测量，不保证其他机器的时间或成绩，也没有按测试结果挑默认配置。

相同默认下，训练评论截断36.55%、验证截断39.2%，验证有效编码字符中 UNK 占0.3926%。这表明前100字会丢掉部分长评论后文，分析时应回看原文。默认校准证据保存在 gitignored 的 `outputs/cpu_validation/`（本任务目录内）。课程原有 torch 2.5.1 环境、GPU、Linux服务器和多人并发尚未在本轮实际验证；实现未为本次任务升级课程依赖。

## 2026-10-17 旧作业与模型验证 历史参考

在上述CPU环境中，00/01/02完整Notebook副本分别从仓库根、RNN目录、文本任务目录执行通过，含内核启动和绘图的壁时分别约6.80/2.54/9.83秒。02同一配置下 RNN/LSTM/GRU/平均Embedding 的验证Accuracy分别为0.648/0.822/0.806/0.850，训练及每轮验证耗时分别约1.099/1.049/2.517/0.286秒；这次简单基线更好，保留全部结果，不把循环模型优势当作预设结论。所有组测试未评价。

学生03原版在基础TODO1按预期停止。仅在gitignored输出副本或内存填教师临时实现，分别验证基础独立闭环、提高换单元、提高masked mean、独立验证分析、独立手写接入和可选梯度检查通过；学生源Notebook及公共手写支架保留TODO。04末尾三练习默认跳过和临时补全后的独立检查也通过。执行副本/JSON记录位于实验室 `outputs/text/notebook_validation/` 与 `outputs/text/student_validation/`，教师临时答案不写回学生文件。

已实际查看训练曲线、验证混淆矩阵、四模型比较图、LSTM有效字符h/c状态图，核对中文、轴、色阶和有效位置。测试关闭时CSV对应列确实留空。本轮未测GPU、Linux服务器、多人并发或课程torch 2.5.1，不能从本机时间推算课堂所有机器的表现。
