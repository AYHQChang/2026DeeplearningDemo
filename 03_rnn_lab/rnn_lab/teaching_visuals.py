"""课堂图解：人工小例子标明来源，实验图从当前结果计算，不触发测试评价。"""

import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

from .plots import configure_chinese_font
from .text_data import encode_text
from .text_engine import predict_text


BLUE, ORANGE, GREEN = "#3175a5", "#d98232", "#258575"


def _figure(*args, **kwargs):
    configure_chinese_font()
    return plt.subplots(*args, constrained_layout=True, **kwargs)


def _card(ax, x, y, width, height, label, color=BLUE, fontsize=11):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012",
                               facecolor=color, alpha=.10, edgecolor=color, linewidth=1.5))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            fontsize=fontsize, linespacing=1.7, color="#203747")


def _arrow(ax, start, end, color=BLUE):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                color=color, linewidth=1.6))


def plot_text_pipeline():
    """不依赖数据与模型的任务地图；文本是教师人工例句。"""
    fig, ax = _figure(figsize=(13, 5.4))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    cards = [(.02, .60, "评论文本\n“房间干净，但隔音不好”\n教师人工例句"),
             (.35, .60, "字符 → 编号\n词表只由训练集建立\n整数 token_ids [B,T]"),
             (.68, .60, "查 Embedding 表\n一个编号取出 E 个数\n浮点向量 [B,T,E]"),
             (.68, .13, "逐字更新状态\n本字向量 + 上一步状态\n循环输出 [B,T,H]"),
             (.35, .13, "读取整句话的表示\n有效句尾 / 排除 PAD 的平均\n句子向量 [B,H]"),
             (.02, .13, "Linear → 两个分数\nsoftmax → 正 / 负概率\nlogits [B,2]")]
    for x, y, label in cards:
        _card(ax, x, y, .28, .27, label)
    for start, end in [((.30,.735),(.35,.735)), ((.63,.735),(.68,.735)),
                       ((.82,.60),(.82,.40)), ((.68,.265),(.63,.265)),
                       ((.35,.265),(.30,.265))]:
        _arrow(ax, start, end)
    ax.text(.5, .98, "一条评论怎样变成一个情感判断？", ha="center", va="top", fontsize=16)
    ax.text(.5, .015, "B = 一批评论数   T = 当前批次补齐长度   E = 字向量维度   H = 状态维度；编号本身没有情感大小", ha="center", fontsize=11)
    return fig


def plot_data_roles(data, max_length=100):
    """类别计数是数据清单；长度分布仅查看训练集，不查看测试预测。"""
    fig, axes = _figure(1, 3, figsize=(13, 4.1))
    splits = ("train", "val", "test")
    bottom = np.zeros(3)
    for label, name in enumerate(data.class_names):
        counts = np.array([sum(row["label"] == label for row in data.splits[s]) for s in splits])
        bars = axes[0].bar(splits, counts, bottom=bottom, label=name)
        axes[0].bar_label(bars, label_type="center", color="white")
        bottom += counts
    axes[0].set(title="原有划分与类别数量", ylabel="评论条数")
    axes[0].legend(loc="upper right")
    lengths = [encode_text(row["text"], {"<PAD>": 0, "<UNK>": 1}, max_length)["original_length"]
               for row in data.splits["train"]]
    # 对长尾压缩横轴只为显示；截断比例仍由全部训练评论计算。
    display_limit = max(max_length * 3, 200)
    axes[1].hist(np.minimum(lengths, display_limit), bins=25, color=BLUE, alpha=.85)
    axes[1].axvline(max_length, color=ORANGE, linestyle="--", label=f"max_length={max_length}")
    truncated = np.mean(np.array(lengths) > max_length)
    axes[1].set(title=f"训练集长度：{truncated:.1%} 超过上限", xlabel=f"清洗后字符数（≥{display_limit} 合入末端）", ylabel="评论条数")
    axes[1].legend()
    ax = axes[2]; ax.axis("off"); ax.set(xlim=(0,1), ylim=(0,1))
    for y, title, color in [(.72,"train：建立词表、更新参数",BLUE),
                             (.40,"val：看曲线、比较模型、分析错例",GREEN),
                             (.08,"test：方案固定后，手动开启一次评价",ORANGE)]:
        _card(ax,.01,y,.96,.20,title,color,10)
    fig.suptitle("先看数据，再训练｜计数不是模型成绩；本图不运行测试评价", fontsize=14)
    return fig


def plot_token_encoding(vocab, text="房间干净，但隔音不好。", max_length=12):
    info = encode_text(text, vocab, max_length)
    chars, ids = info["characters"], info["token_ids"]
    fig, ax = _figure(figsize=(max(8, len(ids)*.55), 3))
    types = np.array([[2 if value == 1 else 1 for value in ids]] * 2)
    ax.imshow(types, cmap=ListedColormap(["#e5e7eb", "#d9eaf5", "#f7d6ab"]), vmin=0, vmax=2, aspect="auto")
    for j, (char, token) in enumerate(zip(chars, ids)):
        ax.text(j, 0, "空格" if char == " " else char, ha="center", va="center", fontsize=15)
        ax.text(j, 1, str(token), ha="center", va="center", fontsize=12)
    ax.set(yticks=[0,1], yticklabels=["有效字符", "词表编号"], xticks=range(len(ids)),
           xticklabels=range(1,len(ids)+1), xlabel="编码位置（从 1 开始）；橙色代表 UNK，编号不是情感分数")
    ax.set_title(f"教师例句 → 编码｜原长 {info['original_length']}，有效长 {info['length']}，截断={info['truncated']}，UNK={info['unknown_count']}")
    return fig


def plot_padding_demo(vocab, texts=("很好", "隔音不太好", "服务很好但是房间太吵"), max_length=12):
    infos = [encode_text(text, vocab, max_length) for text in texts]
    width = max(info["length"] for info in infos)
    grid = np.zeros((len(infos), width), dtype=int)
    fig, ax = _figure(figsize=(max(9,width*.6), max(3.6,len(infos)*.85)))
    for i, info in enumerate(infos):
        for j in range(width):
            if j < info["length"]:
                grid[i,j] = 2 if info["token_ids"][j] == 1 else 1
                label = f"{info['characters'][j]}\n{info['token_ids'][j]}"
            else:
                label = "PAD\n0"
            ax.text(j,i,label,ha="center",va="center",fontsize=10)
        ax.plot(info["length"]-.5,i,"<",color=GREEN,markersize=12,clip_on=False)
    ax.imshow(grid, cmap=ListedColormap(["#e5e7eb","#d9eaf5","#f7d6ab"]), vmin=0,vmax=2,aspect="auto")
    ax.set(xticks=range(width),xticklabels=range(1,width+1),yticks=range(len(infos)),
           yticklabels=[f"样本 {i}｜length={info['length']}" for i,info in enumerate(infos)],
           xlabel="同一列 = 同一个时间位置；绿色标记在有效句尾之后",ylabel="B：各样本独立")
    ax.set_title(f"教师人工短句｜补齐后 [B,T]=[{len(infos)},{width}]；PAD 占位置，不是句中真实字符")
    return fig


def plot_embedding_vectors(embedding, vocab, text="好不好", title="随机初始化的 Embedding（尚未训练）"):
    info = encode_text(text, vocab, max_length=12)
    device = embedding.weight.device
    with torch.no_grad():
        values = embedding(torch.tensor(info["token_ids"],device=device)).detach().cpu().numpy()
    shown = values[:, :min(values.shape[1], 12)]
    fig, ax = _figure(figsize=(9,3.3))
    bound = max(float(np.abs(shown).max()),1e-6)
    im = ax.imshow(shown, aspect="auto",cmap="RdBu_r",vmin=-bound,vmax=bound)
    if shown.size <= 80:
        for i,j in np.ndindex(shown.shape):
            ax.text(j,i,f"{shown[i,j]:.2f}",ha="center",va="center",fontsize=10,
                    color="white" if abs(shown[i,j]) > .65*bound else "#152536")
    ax.set(yticks=range(len(info["characters"])),
           yticklabels=[f"位置 {i+1}：{char} → ID {token}" for i,(char,token) in enumerate(zip(info["characters"],info["token_ids"]))],
           xticks=range(shown.shape[1]),xticklabels=range(shown.shape[1]),xlabel=f"向量坐标（共 E={values.shape[1]}，展示前 {shown.shape[1]} 维）",title=title)
    fig.colorbar(im,ax=ax,label="表中实际权重；红正蓝负")
    return fig


def plot_readout_demo(hidden, lengths):
    """显示给定教学张量；显式区分有效读出与故意包含 PAD 的反例。"""
    from .text_model import valid_last, masked_mean
    values = hidden.detach().cpu().numpy()
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=hidden.device)
    last = valid_last(hidden,lengths).detach().cpu().numpy()
    mean = masked_mean(hidden,lengths).detach().cpu().numpy()
    fig, axes = _figure(len(values),2,figsize=(12,3*len(values)),squeeze=False)
    bound = max(float(np.abs(values).max()),1e-6)
    for i,(left,right) in enumerate(axes):
        im=left.imshow(values[i].T,aspect="auto",cmap="RdBu_r",vmin=-bound,vmax=bound)
        for t in range(values.shape[1]):
            for unit in range(values.shape[2]):
                left.text(t,unit,f"{values[i,t,unit]:g}",ha="center",va="center",
                          color="white" if abs(values[i,t,unit]) > .6*bound else "#152536")
        length=int(lengths[i]); end=values.shape[1]-.5
        if length < values.shape[1]:
            left.axvspan(length-.5,end,facecolor="gray",alpha=.20,hatch="//")
        left.set(title=f"样本 {i}：length={length}；斜线区来自补齐位置",xlabel="时间位置",ylabel="隐藏单元",
                 xticks=range(values.shape[1]),xticklabels=range(1,values.shape[1]+1),yticks=range(values.shape[2]))
        rows=np.stack([last[i],values[i,-1],mean[i],values[i].mean(axis=0)])
        right.axis("off")
        table=right.table(cellText=[[name, np.array2string(row,precision=2)] for name,row in zip(
            ["有效 last", "物理最后一格（反例）", "masked mean", "全 T 平均（反例）"],rows)],
            colLabels=["读取方式", "结果向量"],loc="center",cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1,1.8)
        right.set_title("没有 PAD 时反例可能碰巧相同；有 PAD 才能暴露问题")
    fig.suptitle("人工隐藏状态示意｜padding_idx=0 不会让循环层自动跳过补齐位置",fontsize=14)
    return fig


def plot_review_predictions(result, texts):
    predictions=[predict_text(result,text) for text in texts]
    fig, axes=_figure(len(texts),1,figsize=(11,max(3,2.1*len(texts))),squeeze=False)
    for ax,text,pred in zip(axes[:,0],texts,predictions):
        names=list(pred["probabilities"]); probs=list(pred["probabilities"].values())
        bars=ax.barh(names,probs,color=[ORANGE,BLUE][:len(names)])
        ax.bar_label(bars,labels=[f"{p:.1%}" for p in probs],padding=3)
        ax.set_xlim(0,1.16)
        shown=text if text.strip() else "[空白输入：单 UNK 哨兵，不是有效评论]"
        ax.set_title("\n".join(textwrap.wrap(shown,46))+f"\n预测={pred['predicted_class']}；有效长={pred['length']}，UNK={pred['unknown_count']}，截断={pred['truncated']}",fontsize=11,loc="left")
        ax.set_xlabel("模型 softmax 概率（未校准，不表示正确概率）")
    fig.suptitle("教师人工探针句｜来自当前训练模型的实际输出；不计入验证成绩",fontsize=14)
    return fig


def plot_learning_comparison(results):
    fig,axes=_figure(1,2,figsize=(12,4))
    for result in results:
        hist=result["history"]; label=result["config"].cell_type
        steps=range(1,len(hist["val_loss"])+1)
        axes[0].plot(steps,hist["val_loss"],"o-",label=label)
        axes[1].plot(steps,hist["val_accuracy"],"o-",label=label)
    for ax,title in zip(axes,["验证 Loss（越低越好）","验证 Accuracy（越高越好）"]):
        ax.set(title=title,xlabel="训练轮数");ax.grid(alpha=.2);ax.legend()
    axes[1].set_ylim(0,1)
    fig.suptitle("相同数据与预算下的学习轨迹｜一次种子的观察，不能推出某种单元总是更好")
    return fig


def plot_validation_groups(result):
    metrics=result["val_metrics"]
    records={row["id"]:row for row in result["data"].splits["val"]}
    names=["未截断","被截断","无 UNK","有 UNK"]
    count=np.zeros(4,dtype=int); errors=np.zeros(4,dtype=int)
    lengths=[]; correct=[]
    for index,record_id in enumerate(metrics["ids"]):
        record=records[record_id]
        info=encode_text(record["text"],result["vocab"],result["config"].max_length)
        wrong=int(metrics["predictions"][index]) != record["label"]
        groups=[int(info["truncated"]), 2+int(info["unknown_count"]>0)]
        for group in groups:
            count[group]+=1; errors[group]+=int(wrong)
        lengths.append(info["original_length"]); correct.append(not wrong)
    rate=np.divide(errors,count,out=np.full(4,np.nan),where=count>0)
    fig,axes=_figure(1,2,figsize=(12,4))
    bars=axes[0].bar(names,np.nan_to_num(rate),color=[BLUE,ORANGE,BLUE,ORANGE])
    axes[0].bar_label(bars,labels=[f"{e}/{n}\n{r:.1%}" if n else "无样本" for e,n,r in zip(errors,count,rate)],padding=3)
    axes[0].set(title="验证错误率与分母（两个分组维度会重叠）",ylabel="错误数 / 本组样本数",ylim=(0,1))
    lengths=np.array(lengths); correct=np.array(correct)
    cap=max(result["config"].max_length*3,200); bins=np.linspace(0,cap,21)
    for mask,label,color in [(correct,"正确",BLUE),(~correct,"错误",ORANGE)]:
        axes[1].hist(np.minimum(lengths[mask],cap),bins=bins,alpha=.6,label=label,color=color)
    axes[1].axvline(result["config"].max_length,color="black",linestyle="--",label="截断上限")
    axes[1].set(title="按原始长度看错例在哪里",xlabel=f"清洗后字符数（≥{cap} 合入末端）",ylabel="样本数");axes[1].legend()
    fig.suptitle(f"{result['config'].cell_type}｜验证诊断：相关现象不等于截断或 UNK 导致了错误")
    return fig


def plot_prefix_predictions(result, text="房间很干净，但是隔音太差了。"):
    from .text_observation import observe_text_recurrence
    trace=observe_text_recurrence(result,text)
    chars=trace["characters"]
    if trace["empty"]:
        raise ValueError("逐字观察需要非空文本。")
    # 每个前缀都按完整推断路径计算；mean 读出同样成立。
    prefixes=[predict_text(result,"".join(chars[:i+1])) for i in range(len(chars))]
    fig,axes=_figure(2,1,figsize=(max(11,len(chars)*.43),6),sharex=True)
    x=np.arange(len(chars))
    for name in result["data"].class_names:
        axes[0].plot(x,[p["probabilities"][name] for p in prefixes],"o-",label=name)
    axes[0].set(ylim=(0,1),ylabel="前缀的模型概率",title="读到这里就暂时停止：分类头会输出什么？");axes[0].legend();axes[0].grid(alpha=.2)
    h=trace["hidden_states"].numpy()
    delta=np.linalg.norm(np.diff(np.vstack([np.zeros((1,h.shape[1])),h]),axis=0),axis=1)
    axes[1].bar(x,delta,color=GREEN)
    axes[1].set(ylabel="相邻 h 的欧氏距离",xlabel="教师人工例句：当前位置及其之前的字符",xticks=x,
                xticklabels=[f"{i+1}\n{c}" for i,c in enumerate(chars)],title="状态变化幅度：并非词的重要性、注意力或因果解释")
    fig.suptitle("逐字读入的实际模型观察｜训练标签对应整句，残缺前缀可能没有确定情感",fontsize=14)
    return fig


def plot_assignment_route(kind="text"):
    is_text=kind=="text"
    labels=(["基础 A\n词表 + 有效句尾 forward\n训练 / 曲线 / 验证错例", "提高 B\n只改单元或读出\n控制其他设置", "进阶 C\n先用验证证据提出假设\n再改一个因素", "挑战 D\n手写普通 RNN\n先逐步核对，再训练"] if is_text else
            ["基础 A\n读图 + 逐行转换 + forward\n运行一组完整实验", "提高 B / C\nB 改门控单元\nC 在 B 上改 mean", "进阶 D\n在 B 上改二维分块\n独立坐标 + 逆变换", "挑战 E\n手写普通 RNN\n同设置对照 A"])
    fig,ax=_figure(figsize=(12,4.6));ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
    _card(ax,.02,.28,.24,.42,labels[0],BLUE)
    for i,y in enumerate([.72,.40,.08]):
        _card(ax,.52,y,.45,.21,labels[i+1],GREEN if i<2 else ORANGE,11)
        _arrow(ax,(.26,.49),(.52,y+.105))
    ax.text(.5,.99,"文本：推荐主线作业" if is_text else "图片：选做迁移扩展",ha="center",va="top",fontsize=16)
    ax.text(.27,.05,"先完成基础；提高、进阶、挑战按兴趣选做\n图片的 C/D 需要 B，E 只需要 A",ha="center",fontsize=10)
    if is_text:
        # 文本各分支只依赖基础，不显示图片依赖说明。
        ax.texts[-1].set_text("先完成基础，再选一项扩展\n三个扩展分支彼此独立；不按准确率划等级")
    return fig


def plot_tensor_shapes(batch_size=3,steps=5,features=6,hidden_size=8,layers=2):
    fig,axes=_figure(1,2,figsize=(13,4.8))
    ax=axes[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
    labels=[f"输入\n[B,T,F] = [{batch_size},{steps},{features}]",f"顶层全部输出\n[B,T,H] = [{batch_size},{steps},{hidden_size}]",f"最后状态 h_n\n[L,B,H] = [{layers},{batch_size},{hidden_size}]"]
    for y,label in zip([.72,.40,.08],labels):
        _card(ax,.07,y,.86,.21,label)
    ax.text(.5,.98,"B 是并行样本；F / H 是每个位置的宽度",ha="center",fontsize=12)
    ax=axes[1];ax.axis("off");ax.set(xlim=(-.5,4.7),ylim=(-.75,2.4))
    # 固定画 2 层、4 步作示意；不能将示意节点个数误认为传入的实际形状。
    for layer in range(2):
        for t in range(4):
            ax.text(t,layer,f"h[{layer+1},{t+1}]",ha="center",va="center",bbox=dict(boxstyle="round",fc="#e5f1f6",ec=BLUE))
            if t<3:_arrow(ax,(t+.32,layer),(t+.68,layer))
            if layer==0:_arrow(ax,(t,.22),(t,.78),GREEN)
    ax.text(1.5,-.58,"横向：时间步 T（同层共享参数）",ha="center")
    ax.text(4.25,.5,"纵向\n层数 L\n\n不同层\n各有参数",ha="center")
    ax.text(1.5,1.85,"二维展开示意：2 层 × 4 步\n实际运行形状以左图为准",ha="center",fontsize=12)
    fig.suptitle("时间步 T 与堆叠层数 L 是两回事；batch_first 不改变 h_n 的 [L,B,H]",fontsize=14)
    return fig


def plot_recurrence_arithmetic(x, w_x, w_h, bias, initial, states):
    previous=np.vstack([initial,states[:-1]])
    pieces=[np.asarray(x) @ np.asarray(w_x).T, previous @ np.asarray(w_h).T,
            np.broadcast_to(bias,np.asarray(states).shape), np.asarray(states)]
    fig,axes=_figure(1,4,figsize=(13,4))
    bound=max(float(np.abs(p).max()) for p in pieces) or 1
    for ax,piece,title in zip(axes,pieces,["① 当前输入贡献\nW_x @ x_t","② 历史状态贡献\nW_h @ h_previous","③ 偏置\nb","④ 新状态\ntanh(①+②+③)"]):
        im=ax.imshow(piece,aspect="auto",cmap="RdBu_r",vmin=-bound,vmax=bound)
        for i,j in np.ndindex(piece.shape):
            ax.text(j,i,f"{piece[i,j]:.3f}",ha="center",va="center",fontsize=10,
                    color="white" if abs(piece[i,j]) > .6*bound else "#152536")
        ax.set(title=title,xlabel="隐藏单元",yticks=range(piece.shape[0]),yticklabels=[f"第 {i+1} 步" for i in range(piece.shape[0])],xticks=range(piece.shape[1]))
    fig.colorbar(im,ax=axes.tolist(),label="真实计算值；四图共享色阶")
    fig.suptitle("人工权重的逐步计算｜读每一行：新信息 + 旧信息 + 偏置 → 新状态",fontsize=14)
    return fig


def plot_scores(logits, probabilities, class_names=None):
    logits=torch.as_tensor(logits).detach().cpu().numpy().reshape(-1)
    probabilities=torch.as_tensor(probabilities).detach().cpu().numpy().reshape(-1)
    names=class_names or [f"类别 {i}" for i in range(len(logits))]
    fig,axes=_figure(1,2,figsize=(10,3.7))
    for ax,values,title,color in zip(axes,[logits,probabilities],["Linear 原始分数：logits","softmax：同一行归一化"],[BLUE,GREEN]):
        bars=ax.bar(names,values,color=color);ax.bar_label(bars,fmt="%.3f",padding=3);ax.set_title(title);ax.axhline(0,color="gray",lw=.7);ax.margins(y=.25)
    axes[1].set_ylim(0,1.16)
    fig.suptitle(f"同一次前向传播｜argmax={int(logits.argmax())}；概率之和={probabilities.sum():.3f}")
    return fig


def plot_cell_memory_map():
    fig,axes=_figure(1,3,figsize=(13,4.4))
    for ax,title,top,bottom in zip(axes,["RNN","GRU","LSTM"],
            ["h：更新后的状态","h：经过门控制的状态","h：用于输出的隐藏状态"],
            ["tanh(输入贡献 + 历史贡献)\n只有一套状态 h","更新门 / 重置门\n控制保留与候选信息的混合","另有 c：细胞状态\n输入门 / 遗忘门 / 输出门"]):
        ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        _card(ax,.05,.63,.9,.23,top,BLUE,11)
        _card(ax,.05,.12,.9,.31,bottom,GREEN,11)
        _arrow(ax,(.5,.43),(.5,.63));ax.set_title(title,fontsize=15)
    fig.suptitle("结构对照示意（非完整门公式）｜门控改变更新方式，不保证本次验证一定更高",fontsize=14)
    return fig


def plot_image_sequence_preview():
    """只画空间对应，不提供学生转换/逆变换函数的实现。"""
    fig,axes=_figure(1,3,figsize=(12,4))
    matrix=np.arange(48).reshape(6,8)
    for ax,title in zip(axes,["教师编号示意：6 行 × 8 列","逐行：T=6，F=8","2×2 分块：T=12，F=4"]):
        ax.imshow(matrix,cmap="Blues",alpha=.6)
        for i,j in np.ndindex(matrix.shape):ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=9)
        ax.set(title=title,xlabel="原图列",ylabel="原图行",xticks=range(8),yticks=range(6))
    axes[1].axhspan(1.5,2.5,facecolor=ORANGE,alpha=.35)
    from matplotlib.patches import Rectangle
    axes[2].add_patch(Rectangle((1.5,1.5),2,2,fill=False,edgecolor=ORANGE,linewidth=3))
    fig.suptitle("图片扩展：顺序由读取规则规定｜橙色框是一条时间步，真实训练仍使用 Fashion 图片",fontsize=14)
    return fig


def plot_text_levels():
    """新版文本三级；旧A–D图继续用于历史参考，不重解释旧图。"""
    fig, axes = _figure(1, 3, figsize=(13, 4.6))
    items = [
        ("第一级 运行与调参", "完整普通 RNN", "运行基线 → 修改一个参数\n保留两组曲线与验证结果", BLUE),
        ("第二级 补全 LSTM", "构造和 forward 的三处 TODO", "按形状提示补齐 → 小批次核对\n通过后训练，与 RNN 比较", GREEN),
        ("第三级 自主搭建 GRU", "给接口、步骤和检查", "自行写完整模型 → 接入训练\n解释结构选择与比较条件", ORANGE),
    ]
    for ax, (title, provided, task, color) in zip(axes, items):
        ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1))
        _card(ax, .03, .63, .94, .23, provided, color, 12)
        _card(ax, .03, .13, .94, .32, task, color, 11)
        _arrow(ax, (.5, .63), (.5, .45), color)
        ax.set_title(title, fontsize=14)
    fig.suptitle("同一文本任务，逐步增加实现工作｜后二级独立开启，不要求准确率逐级提高", fontsize=14)
    return fig


def plot_text_model_steps(cell_type, embedding_dim=32, hidden_size=32):
    """接口/模块图，不提供LSTM/GRU的可复制分类器答案。"""
    if cell_type not in {"rnn", "lstm", "gru"}:
        raise ValueError("请选择 rnn、lstm 或 gru。")
    fig, ax = _figure(figsize=(13, 4.6))
    ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1))
    captions = [f"Embedding\n整数 [B,T]\n→ 浮点 [B,T,{embedding_dim}]",
                f"{cell_type.upper()}\n逐步状态更新\n→ [B,T,{hidden_size}]",
                f"句子读出\n排除右侧 PAD\n→ [B,{hidden_size}]",
                "Linear\n每句两个原始分数\n→ logits [B,2]"]
    for i, label in enumerate(captions):
        x = .025 + i * .25
        _card(ax, x, .34, .20, .37, label, GREEN if cell_type == "lstm" else BLUE, 11)
        if i < 3:
            _arrow(ax, (x + .20, .525), (x + .25, .525))
    notes = {"rnn": "完整实现已提供；先预测每步形状，再运行与调参。",
             "lstm": "TODO 1 构造循环层    TODO 2 连接前向与循环输出    TODO 3 有效句尾及分类\nLSTM另返回(h_n,c_n)，二者不是全部逐时刻输出。",
             "gru": "你来决定 E/H/L 与读出；自己定义各层并连接 forward。\n允许使用 nn.GRU；先通过小批次检查，再训练。"}
    ax.text(.5, .14, notes[cell_type], ha="center", va="center", fontsize=11, linespacing=1.7)
    ax.text(.5, .95, "从输入到分类：每个方框都应能说明形状和作用", ha="center", va="top", fontsize=16)
    return fig
