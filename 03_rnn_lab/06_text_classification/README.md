# 中文评论情感分类 三级实验

这是RNN课堂主线。学生唯一作业入口是[03 三级实践](03_自主修改与挑战.ipynb)：一级完整RNN调参，二级补LSTM，三级自行搭GRU。三个等级共用数据和训练工具，不按准确率分档，不要求同时完成图片扩展。

## 学生怎样开始

| 文件 | 现在的用途 |
|---|---|
| [00 情感体验](00_情感分类体验.ipynb) | 已完成GRU的教师参考结果，保留曲线/概率等旧图，参考重算默认关闭 |
| [01 编码讲解](01_文字如何变成序列.ipynb) | 字符/ID、PAD、Embedding、有效句尾图解，可直接运行 |
| [02 模型比较](02_训练与模型对比.ipynb) | 四模型、状态、前缀和错例的历史参考，重算默认关闭 |
| [03 三级作业](03_自主修改与挑战.ipynb) | 统一学生入口，默认完整运行一级，后二级独立开启 |

一级先运行普通RNN基线，再在集中参数区只改一个因素。默认示范将learning_rate从0.003改为0.0015，自动保留两组结果；这些是运行起点，不是更优参数承诺。E=32、H=32、1层、max_length=100、batch=64、6轮、seed=42、CPU单线程。

二级在StudentLSTMClassifier中补三处：构造循环层、连接循环前向、有效句尾及分类。每处有形状/功能说明和常见错误，完成后先查输出形状、有限值、增加右PAD和长短句混批不变性，再训练。三级StudentGRUClassifier只有接口和设计提示，允许torch.nn.GRU，自己定义Embedding、循环层、读出和分类器。E/H/L/读出与实际模型保持一致。

RUN_LEVEL2、RUN_LEVEL3默认False；关闭分支只说明跳过，不创建未完成模型。第三级不依赖第二级。完整参考模型仍在../rnn_lab/text_model.py，学生训练经text_course.py显式提交自己的模型，不能通过省略model、继承完整TextClassifier或修改cell_type自动获得答案。接口检查属于功能检查，不是自动评分。

使用现有课程Python环境，不需要Paddle、torchtext、在线词向量、预训练模型或pip安装rnn_lab。支持仓库根、03_rnn_lab和本目录作为工作目录。修改类、词表或配置后重启内核从头运行。每次只运行一个训练任务，CPU单线程、workers=0。

## 参考输出如何保留

四本调整前的完整文件保存在[reference说明](reference/README.md)，其中20261017/manifest.json记录原文件SHA256。00/02当前输出块保持原样，页首标为教师参考；03原有两幅图保存在末尾历史区。原始数据、CSV、模型、图和旧输出目录不动。旧任务A/B/C/D不是新等级编号。

RUN_TEACHER_REFERENCE=False关闭00/02的参考计算；教师重新开启时按原模型重新运行并生成新目录。参考关闭不会留下未定义的result变量调用。归档保留原文件字节，因此其中旧相对链接仍按原任务目录解释，当前导航用本README。

学生报告中的实验结果来自自己的运行，不能把历史GRU、LSTM或四模型成绩改名为本次RNN成绩。人工探针只作解释和额外预测，不能计入正式验证成绩。

## 报告与结果提交

使用[独立RNN Word模板](../../docs/03_RNN模型组合探索实验报告模板.docx)。只交一份：基本信息；一级基线/调参及图；二级补全与检查；三级结构设计与结果；总结。未开展部分注明即可。仅提交一份Word报告；个人Notebook、CSV和必要图留在个人目录备查，不把全量代码和所有输出复制进Word。

新输出在实验室outputs/text/three_levels/时间_uuid/。每组保留配置、模型、预处理、history、CSV和图；汇总CSV保留本轮所有真正完成组。正式测试默认关闭，固定方案后只在最终格手动开启。记录的implementation区分提供的RNN、学生LSTM与学生GRU；未评价的测试列为空。

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
