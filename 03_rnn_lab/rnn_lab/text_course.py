"""三级文本实验入口。完整门控模型留在 text_model 作为教师参考。

一级提供完整普通 RNN；二/三级只训练调用方明确提交的学生模型。
"""
from dataclasses import replace

import torch
from torch import nn

from .observation import evaluating
from .text_engine import TextExperimentConfig, run_text_experiment
from .text_model import TextClassifier


TUNABLE_PARAMETERS = {"learning_rate", "hidden_size", "epochs", "batch_size", "max_length",
                      "embedding_dim", "num_layers"}


def make_tuned_config(base: TextExperimentConfig, parameter: str, value):
    """一次改变一个参数；其他配置含数据边界与随机种子保持。"""
    if base.cell_type != "rnn" or base.readout != "last":
        raise ValueError("第一级固定普通 RNN 与有效 last；模型比较在第二/三级完成。")
    if parameter not in TUNABLE_PARAMETERS:
        raise ValueError("请选择参数区中的学习率、H、轮数、batch、最大长度、E或层数。")
    if value == getattr(base, parameter):
        raise ValueError("调参值与基线相同，请选择一个不同值。")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("参数应为正数。")
    if parameter != "learning_rate" and not isinstance(value, int):
        raise ValueError("维度、长度、层数、batch和轮数应为正整数。")
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
    device = parameters[0].device
    ids = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.int64, device=device)
    lengths = torch.tensor([2, 4], dtype=torch.int64, device=device)
    with evaluating(model), torch.no_grad():
        logits = model(ids, lengths)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (2, n_classes):
            raise ValueError(f"forward 应返回 [B,K]=[2,{n_classes}] logits。")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("logits 含 NaN/Inf，请检查前向计算。")
        padded = model(torch.nn.functional.pad(ids, (0, 3)), lengths)
        alone = model(ids[:1, :2], lengths[:1])
        try:
            torch.testing.assert_close(logits, padded, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(logits[:1], alone, atol=1e-6, rtol=1e-5)
        except AssertionError as exc:
            raise ValueError("增加右侧 PAD 或混合不同长度改变了有效句子结果，请检查读出位置/掩码。") from exc
    return {"output_shape": tuple(logits.shape), "finite": True,
            "right_padding_invariant": True, "mixed_batch_invariant": True,
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
