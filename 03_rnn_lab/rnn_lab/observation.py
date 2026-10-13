"""真实前向传播的形状和逐时刻状态观察，不替学生实现分类器。"""

from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def evaluating(model: nn.Module):
    """临时进入评估模式，结束后逐模块恢复原状态（包括混合模式）。"""

    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def observe_recurrence(model: nn.Module, x: torch.Tensor, index: int = 0) -> dict:
    """观察实际分类前向，并核对同一循环层逐时刻重放的 h/c。

    模型需具有 ``recurrent`` 属性（单向、batch_first 的 RNN/LSTM/GRU）。
    可选的 ``classifier`` 属性用于记录真实分类层输入，而非假定最后时刻读取。
    仅在 eval/no_grad 下观察；不会移动模型、写梯度或改变训练状态。
    图中 h_t/c_t 展示顶层，完整各层轨迹保存在 all_hidden_states/all_cell_states。
    """

    recurrent = getattr(model, "recurrent", None)
    if not isinstance(recurrent, (nn.RNN, nn.LSTM, nn.GRU)):
        raise ValueError("观察模型需要 recurrent 属性，类型为 nn.RNN、nn.LSTM 或 nn.GRU。")
    if not recurrent.batch_first or recurrent.bidirectional:
        raise ValueError("当前教学观察支持 batch_first=True 的单向循环层。")
    if isinstance(recurrent, nn.LSTM) and recurrent.proj_size:
        raise ValueError("当前教学观察暂不支持带 proj_size 的 LSTM。")
    if x.ndim != 3 or min(x.shape) < 1:
        raise ValueError("请输入非空三维张量 [B,T,F]；图片需先由学生转换成序列。")
    if x.shape[2] != recurrent.input_size:
        raise ValueError(f"每步特征数 F={x.shape[2]}，但循环层 input_size={recurrent.input_size}。")
    if not 0 <= index < len(x):
        raise ValueError("index 必须指向批量中的一个样本。")
    device = next(model.parameters()).device
    sample_batch = x.detach().to(device)
    calls = []
    readouts = []

    def capture_recurrent(module, inputs, output):
        calls.append((inputs, output))

    def capture_classifier(module, inputs, output):
        readouts.append((tuple(inputs[0].shape), tuple(output.shape)))

    handles = [recurrent.register_forward_hook(capture_recurrent)]
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Module):
        handles.append(classifier.register_forward_hook(capture_classifier))
    try:
        with evaluating(model), torch.no_grad():
            logits = model(sample_batch)
    finally:
        for handle in handles:
            handle.remove()
    if len(calls) != 1:
        raise ValueError("当前观察需要模型在一次 forward 中调用 recurrent 恰好一次。")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or len(logits) != len(x):
        raise ValueError("分类模型应返回 [B,K] logits，请检查学生 forward 的返回值。")
    recurrent_inputs, (outputs, final_state) = calls[0]
    actual_input = recurrent_inputs[0]
    if actual_input.ndim != 3 or actual_input.shape[:2] != sample_batch.shape[:2]:
        raise ValueError("循环层接收的批量和时间轴与给定输入不同，不能按当前样本索引观察。")
    initial = recurrent_inputs[1] if len(recurrent_inputs) > 1 else None

    def select_state(state):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(part[:, index:index + 1].detach() for part in state)
        return state[:, index:index + 1].detach()

    state = select_state(initial)
    hidden_steps, cell_steps, output_steps = [], [], []
    with evaluating(model), torch.no_grad():
        for step in range(actual_input.shape[1]):
            step_output, state = recurrent(actual_input[index:index + 1, step:step + 1], state)
            output_steps.append(step_output)
            hidden = state[0] if isinstance(state, tuple) else state
            hidden_steps.append(hidden[:, 0].cpu())
            if isinstance(state, tuple):
                cell_steps.append(state[1][:, 0].cpu())
    replayed = torch.cat(output_steps, dim=1)
    expected = outputs[index:index + 1]
    replay_error = float((replayed - expected).abs().max())
    if not torch.allclose(replayed, expected, rtol=1e-4, atol=1e-5):
        raise RuntimeError("逐时刻重放与真实循环输出不一致；此模型不能使用当前观察方式。")
    state_parts = state if isinstance(state, tuple) else (state,)
    final_parts = final_state if isinstance(final_state, tuple) else (final_state,)
    for replay_part, expected_part in zip(state_parts, final_parts):
        if not torch.allclose(replay_part, expected_part[:, index:index + 1], rtol=1e-4, atol=1e-5):
            raise RuntimeError("逐时刻重放的最终 h_n/c_n 与真实前向不一致。")

    h_n = final_parts[0]
    c_n = final_parts[1] if len(final_parts) == 2 else None
    all_hidden = torch.stack(hidden_steps)
    all_cell = torch.stack(cell_steps) if cell_steps else None
    cell_name = recurrent.__class__.__name__
    recurrent_parameters = (
        f"{cell_name}; F={recurrent.input_size}, H={recurrent.hidden_size}, "
        f"层数={recurrent.num_layers}, dropout={recurrent.dropout:g}"
    )
    rows = [
        {"name": "输入序列", "input_shape": tuple(x.shape), "output_shape": tuple(actual_input.shape),
         "parameters": "batch_first=True", "explanation": "B 是样本数，T 是时间步数，F 是每步特征数。"},
        {"name": "循环层全部输出", "input_shape": tuple(actual_input.shape), "output_shape": tuple(outputs.shape),
         "parameters": recurrent_parameters, "explanation": "[B,T,H] 保存顶层每个时刻的输出；时间步间共享权重。"},
        {"name": "最终隐藏状态 h_n", "input_shape": tuple(outputs.shape), "output_shape": tuple(h_n.shape),
         "parameters": "[层数,B,H]", "explanation": "保存各层的最后状态；batch_first 不改变此张量的轴顺序。"},
    ]
    if c_n is not None:
        rows.append({"name": "最终细胞状态 c_n", "input_shape": tuple(outputs.shape), "output_shape": tuple(c_n.shape),
                     "parameters": "LSTM 独有；[层数,B,H]", "explanation": "c 与 h 是不同的状态；c 不要求位于 [-1,1]。"})
    if len(readouts) == 1:
        readout_shape, classifier_shape = readouts[0]
        rows.append({"name": "分类层", "input_shape": readout_shape, "output_shape": classifier_shape,
                     "parameters": str(classifier), "explanation": "真实捕获的分类层输入；读取方式由模型 forward 决定。"})
    rows.extend([
        {"name": "logits → softmax", "input_shape": tuple(logits.shape), "output_shape": tuple(logits.shape),
         "parameters": "dim=1", "explanation": "logits 是未归一化分数；概率沿类别轴求和为 1。"},
        {"name": "argmax 类别", "input_shape": tuple(logits.shape), "output_shape": (len(logits),),
         "parameters": "dim=1", "explanation": "取分数或概率最大的类别索引；训练交叉熵直接接收 logits。"},
    ])
    return {
        "rows": rows, "input_shape": tuple(x.shape), "recurrent_output_shape": tuple(outputs.shape),
        "output_shape": tuple(logits.shape), "h_n_shape": tuple(h_n.shape),
        "c_n_shape": None if c_n is None else tuple(c_n.shape),
        "input": actual_input[index].detach().cpu(), "hidden_states": all_hidden[:, -1],
        "cell_states": None if all_cell is None else all_cell[:, -1],
        "all_hidden_states": all_hidden, "all_cell_states": all_cell,
        "h_n": h_n.detach().cpu(), "c_n": None if c_n is None else c_n.detach().cpu(),
        "logits": logits.detach().cpu(), "probabilities": logits.softmax(dim=1).detach().cpu(),
        "predictions": logits.argmax(dim=1).detach().cpu(), "index": index,
        "cell_type": cell_name, "replay_max_error": replay_error,
    }


def print_model_summary(model: nn.Module, x: torch.Tensor) -> list[dict]:
    """打印真实前向的尺寸、参数和解释，返回可继续处理的表格记录。"""

    rows = observe_recurrence(model, x)["rows"]
    for row in rows:
        print(f"{row['name']}: {row['input_shape']} → {row['output_shape']}")
        print(f"  {row['parameters']}；{row['explanation']}")
    return rows
