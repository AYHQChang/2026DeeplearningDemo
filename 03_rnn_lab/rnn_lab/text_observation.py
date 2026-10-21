"""只观察有效字符：真实前向、逐时刻重放及 LSTM 的 h/c。"""

import torch
from torch import nn

from .observation import evaluating
from .text_data import encode_text


def observe_text_recurrence(result: dict, text: str) -> dict:
    """不搬动模型、不改模式或梯度，临时 hook 在任何退出路径均移除。"""
    model = result["model"]
    recurrent = getattr(model, "recurrent", None)
    if not isinstance(recurrent, (nn.RNN, nn.LSTM, nn.GRU)):
        raise ValueError("状态观察支持标准 RNN/LSTM/GRU；平均字向量没有循环状态。")
    if not recurrent.batch_first or recurrent.bidirectional or getattr(recurrent, "proj_size", 0):
        raise ValueError("观察要求单向 batch_first 循环层，且 LSTM 不使用投影。")
    encoded = encode_text(text, result["vocab"], result["config"].max_length)
    device = next(model.parameters()).device
    ids = torch.tensor([encoded["token_ids"]], dtype=torch.int64, device=device)
    lengths = torch.tensor([encoded["length"]], dtype=torch.int64, device=device)
    calls, readouts = [], []

    def capture_recurrent(module, inputs, output):
        calls.append((inputs, output))

    def capture_classifier(module, inputs, output):
        readouts.append(inputs[0].detach().cpu())

    handles = [recurrent.register_forward_hook(capture_recurrent)]
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Module):
        handles.append(classifier.register_forward_hook(capture_classifier))
    try:
        with evaluating(model), torch.no_grad():
            logits = model(ids, lengths)
    finally:
        for handle in handles:
            handle.remove()
    if len(calls) != 1:
        raise ValueError("实际 forward 应恰好调用一次循环层。")
    inputs, (outputs, final_state) = calls[0]
    embedded = inputs[0]
    if not isinstance(embedded, torch.Tensor) or embedded.shape[:2] != ids.shape:
        raise ValueError("观察要求循环层接收该文本的 [1,T,E] 字向量。")
    state = inputs[1] if len(inputs) > 1 else None
    hidden_steps, cell_steps, replay_outputs = [], [], []
    with evaluating(model), torch.no_grad():
        for index in range(encoded["length"]):
            step_output, state = recurrent(embedded[:, index:index + 1], state)
            replay_outputs.append(step_output)
            hidden_steps.append((state[0] if isinstance(state, tuple) else state)[:, 0].cpu())
            if isinstance(state, tuple):
                cell_steps.append(state[1][:, 0].cpu())
    replayed = torch.cat(replay_outputs, dim=1)
    replay_error = float((replayed - outputs).abs().max())
    if not torch.allclose(replayed, outputs, rtol=1e-4, atol=1e-5):
        raise RuntimeError("逐时刻重放与真实 forward 的输出不一致。")
    actual_parts = final_state if isinstance(final_state, tuple) else (final_state,)
    replay_parts = state if isinstance(state, tuple) else (state,)
    for actual, replay in zip(actual_parts, replay_parts):
        if not torch.allclose(actual, replay, rtol=1e-4, atol=1e-5):
            raise RuntimeError("逐时刻最终状态与真实 h_n/c_n 不一致。")
    all_hidden = torch.stack(hidden_steps)
    all_cell = torch.stack(cell_steps) if cell_steps else None
    return {**encoded, "text": text, "token_ids_tensor": ids.cpu(),
            "embedding": embedded[0].detach().cpu(), "hidden_states": all_hidden[:, -1],
            "cell_states": None if all_cell is None else all_cell[:, -1],
            "all_hidden_states": all_hidden, "all_cell_states": all_cell,
            "h_n": actual_parts[0].detach().cpu(),
            "c_n": actual_parts[1].detach().cpu() if len(actual_parts) == 2 else None,
            "logits": logits.detach().cpu(), "probabilities": logits.softmax(dim=1).detach().cpu(),
            "readout": readouts[0] if len(readouts) == 1 else None,
            "input_shape": tuple(ids.shape), "embedding_shape": tuple(embedded.shape),
            "recurrent_output_shape": tuple(outputs.shape), "cell_type": type(recurrent).__name__,
            "replay_max_error": replay_error}
