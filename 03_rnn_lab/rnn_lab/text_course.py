"""三级文本实验入口。完整门控模型留在 text_model 作为教师参考。

一级提供完整普通 RNN；二/三级只训练调用方明确提交的学生模型。
"""
from dataclasses import replace
from copy import deepcopy
import math

import torch
from torch import nn

from .observation import evaluating
from .text_engine import TextExperimentConfig, run_text_experiment
from .text_model import TextClassifier, valid_last, masked_mean


TUNABLE_PARAMETERS = {"learning_rate", "hidden_size", "epochs", "batch_size", "max_length",
                      "embedding_dim", "num_layers", "optimizer", "weight_decay", "gradient_clip", "dropout"}


def make_tuned_config(base: TextExperimentConfig, parameter: str, value):
    """一次改变一个参数；其他配置含数据边界与随机种子保持。"""
    if base.cell_type != "rnn" or base.readout != "last":
        raise ValueError("第一级固定普通 RNN 与有效 last；模型比较在第二/三级完成。")
    if parameter not in TUNABLE_PARAMETERS:
        raise ValueError("请选择参数表中的单个训练参数；不能通过切换循环单元完成高等级。")
    if value == getattr(base, parameter):
        raise ValueError("调参值与基线相同，请选择一个不同值。")
    if parameter == "optimizer":
        if value not in {"adam", "sgd", "adamw"}:
            raise ValueError("optimizer 为 adam、sgd 或 adamw。")
    elif parameter == "gradient_clip" and value is None:
        pass
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("数值参数必须为有限数。")
        if value < 0 or (value == 0 and parameter not in {"dropout", "weight_decay"}):
            raise ValueError("参数超出允许范围。")
        if parameter in {"hidden_size", "epochs", "batch_size", "max_length", "embedding_dim", "num_layers"} and not isinstance(value, int):
            raise ValueError("维度、长度、层数、batch和轮数应为正整数。")
        if parameter == "dropout" and (value >= 1 or base.num_layers == 1):
            raise ValueError("循环层间 dropout 需小于1；单层不生效，请不要把它作为有效调参。")
    return replace(base, name=f"一级调参｜{parameter}={value}", **{parameter: value})


def check_student_model(model: nn.Module, n_classes: int, cell_type: str) -> dict:
    """用人工 ID 小批次核对接口和变长读出，不训练、不评价正式数据。"""
    if not isinstance(model, nn.Module) or isinstance(model, TextClassifier):
        raise TypeError("请提交自己编写的 nn.Module 分类器，不能直接使用或继承完整教师 TextClassifier。")
    if cell_type not in {"lstm", "gru"}:
        raise ValueError("学生补全/自主设计检查只接受 lstm 或 gru。")
    expected = nn.LSTM if cell_type == "lstm" else nn.GRU
    recurrent = getattr(model, "recurrent", None)
    if not isinstance(recurrent, expected):
        raise TypeError(f"请在 model.recurrent 中定义实际的 {expected.__name__}，检查模型构造 TODO。")
    if not recurrent.batch_first or recurrent.bidirectional:
        raise ValueError("本次统一接口为 batch_first=True 的单向网络。")
    if getattr(model, "cell_type", None) != cell_type:
        raise ValueError(f"请将 model.cell_type 明确设置为 {cell_type!r}。")
    if getattr(model, "readout", None) not in {"last", "mean"}:
        raise ValueError("请用 model.readout 记录 last 或 mean，与实际实现一致。")
    if any(isinstance(module, (nn.Softmax, nn.LogSoftmax)) for module in model.modules()):
        raise ValueError("forward 返回 logits；请把概率变换留在预测展示阶段。")
    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("学生模型没有可训练参数，请完成构造函数。")
    embedding, classifier = getattr(model, "embedding", None), getattr(model, "classifier", None)
    if not isinstance(embedding, nn.Embedding) or not isinstance(classifier, nn.Module):
        raise ValueError("需要 embedding 和 classifier 模块，便于检查计算路径。")
    if embedding.num_embeddings < 2 or embedding.padding_idx != 0:
        raise ValueError("词表需含 PAD/UNK，且 Embedding padding_idx=0。")
    device = parameters[0].device
    ids = torch.tensor([[1, 2, 0, 0, 0], [3, 2, 4, 1, 5]], dtype=torch.int64, device=device)
    ids = torch.where(ids == 0, ids, (ids - 1) % (embedding.num_embeddings - 1) + 1)
    lengths = torch.tensor([2, 5], dtype=torch.int64, device=device)
    before = deepcopy(model.state_dict())
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), evaluating(model):
            with torch.no_grad():
                logits = model(ids, lengths)
                if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (2, n_classes):
                    raise ValueError("forward 应返回 [B,K]=[2,K] logits。")
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("logits 含 NaN/Inf，请检查前向计算。")
                try:
                    torch.testing.assert_close(logits, model(torch.nn.functional.pad(ids, (0, 3)), lengths), atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(logits[:1], model(ids[:1, :2], lengths[:1]), atol=1e-6, rtol=1e-5)
                except AssertionError as exc:
                    raise ValueError("增加右侧 PAD 或混合不同长度改变了有效句子结果，请检查读出位置/掩码。") from exc
                # 用不同时间位置具有不同数值的状态替换循环输出，核对读出的语义。
                probe = torch.arange(2 * 5 * recurrent.hidden_size, device=device, dtype=parameters[0].dtype).reshape(2, 5, recurrent.hidden_size) / 17
                captured = []
                def replace_states(module, args, output):
                    return (probe, output[1])
                def capture_sentence(module, args):
                    captured.append(args[0].detach().clone())
                handles = [recurrent.register_forward_hook(replace_states), classifier.register_forward_pre_hook(capture_sentence)]
                try:
                    probe_logits = model(ids, lengths)
                finally:
                    for handle in handles:
                        handle.remove()
                expected_sentence = (valid_last if model.readout == "last" else masked_mean)(probe, lengths)
                if len(captured) != 1:
                    raise ValueError("请把循环读出直接送入 classifier 模块一次。")
                try:
                    torch.testing.assert_close(captured[0], expected_sentence)
                    torch.testing.assert_close(probe_logits, classifier(expected_sentence))
                except AssertionError as exc:
                    raise ValueError("读出语义不符：请核对有效句尾/排除PAD的平均，并直接返回 classifier 的 logits。") from exc
            # autograd.grad 不写入已有 .grad；eval 关闭 dropout，禁用 cuDNN 使 GPU eval RNN 也可反传。
            trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.cross_entropy(model(ids, lengths), torch.tensor([0, 1], device=device))
                if not loss.requires_grad:
                    raise ValueError("输出已脱离计算图；请勿在 forward 中 detach 或 no_grad。")
                gradients = torch.autograd.grad(loss, [p for _, p in trainable], allow_unused=True)
            for prefix in ("embedding.", "recurrent.", "classifier."):
                group = [g for (name, _), g in zip(trainable, gradients) if name.startswith(prefix)]
                if not group or any(g is None or not bool(torch.isfinite(g).all()) for g in group):
                    raise ValueError(f"{prefix[:-1]} 未完整接入有效反向计算，请检查 detach、未使用层或冻结参数。")
    finally:
        model.load_state_dict(before)
    return {"output_shape": tuple(logits.shape), "finite": True,
            "right_padding_invariant": True, "mixed_batch_invariant": True,
            "readout_semantics": True, "gradient_connected": True,
            "recurrent": type(recurrent).__name__}


def run_course_experiment(config: TextExperimentConfig, data, *, level: int,
                          vocab=None, model: nn.Module | None = None) -> dict:
    """学生入口始终只看验证成绩，绝不为缺失学生模型补上教师模型。"""
    if level not in {1, 2, 3}:
        raise ValueError("文本实验只有第1、2、3级。")
    expected = {1: "rnn", 2: "lstm", 3: "gru"}[level]
    if config.cell_type != expected:
        raise ValueError(f"第{level}级固定 {expected}；不能通过切换 cell_type 跳过学生实现。")
    if level == 1:
        if model is not None or config.readout != "last":
            raise ValueError("第一级使用提供的完整普通 RNN 与有效 last；自定义模型在后两级提交。")
    else:
        if model is None:
            raise NotImplementedError(f"第{level}级尚未提供学生模型；请完成模型类，再显式传入 model=你的模型。")
        check_student_model(model, data.n_classes, expected)
        embedding = getattr(model, "embedding", None)
        recurrent = model.recurrent
        actual = (getattr(embedding, "embedding_dim", None), recurrent.hidden_size,
                  recurrent.num_layers, getattr(model, "readout", None))
        recorded = (config.embedding_dim, config.hidden_size, config.num_layers, config.readout)
        if actual != recorded:
            raise ValueError("模型实际 E/H/L/读出与 config 不一致，请同步设计参数，避免CSV记录失真。")
        expected_dropout = config.dropout if config.num_layers > 1 else 0.0
        if recurrent.dropout != expected_dropout:
            raise ValueError("循环层实际 dropout 与配置不一致；单层实际层间dropout应为0。")
    # 一级仅走普通RNN分支；二/三级已检查非None，并明确传入，不会触发默认构造。
    result = run_text_experiment(config, data, vocab=vocab, model=model, evaluate_test=False)
    result["implementation"] = {1: "level1_provided_rnn", 2: "level2_student_lstm",
                                3: "level3_student_gru"}[level]
    result["course_level"] = level
    return result


LEVEL_COEFFICIENTS = {1: 0.8, 2: 0.9, 3: 1.0}


def course_score(result: dict) -> dict:
    """已测试模型的折算分；等级完成与独立实现仍由教师验收。不会触发测试。"""
    from .text_engine import _check_cached_test
    _check_cached_test(result)
    level = result.get("course_level")
    if level not in LEVEL_COEFFICIENTS or result.get("metrics") is None:
        raise ValueError("请先固定已完成等级的模型，再显式完成最终测试。")
    accuracy = float(result["metrics"]["accuracy"])
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("测试 Accuracy 必须是 0到1 的比例。")
    coefficient = LEVEL_COEFFICIENTS[level]
    return {"course_level": level, "test_accuracy": accuracy, "coefficient": coefficient,
            "provisional_score": 100 * accuracy * coefficient,
            "acceptance": "待教师核验等级完成条件与独立实现"}


def check_design_pair(first, second):
    """三级同模型单因素设计对照；允许读出、H、E或层数之一，预算保持一致。"""
    ignored = {"name"}
    changes = [key for key in first.__dataclass_fields__ if key not in ignored
               and getattr(first, key) != getattr(second, key)]
    if first.cell_type != "gru" or second.cell_type != "gru" or len(changes) != 1 or changes[0] not in {"readout", "hidden_size", "embedding_dim", "num_layers"}:
        raise ValueError("三级对照需保持GRU及训练条件，仅改变读出、H、E或层数中的一项。")
    return changes[0]
