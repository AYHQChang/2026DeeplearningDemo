"""两条应用路线共用的手写 RNN 练习支架；递推与时间展开留给使用者。"""
import math
import torch
from torch import nn
from .observation import evaluating

_NAMES = ('weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0')

class StudentRNN(nn.Module):
    """单层、单向、tanh、batch_first；接口对齐 nn.RNN。

    修改本文件的 step 和 forward 后，两条应用路线可复用同一份实现。
    禁止用 nn.RNN / nn.RNNCell 替代递推。初态 h0 为 [1,B,H]，默认零。
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        if not isinstance(input_size, int) or not isinstance(hidden_size, int) or min(input_size, hidden_size) < 1:
            raise ValueError('input_size 和 hidden_size 必须是正整数。')
        self.input_size, self.hidden_size = input_size, hidden_size
        self.num_layers, self.batch_first, self.bidirectional = 1, True, False
        self.dropout, self.nonlinearity = 0.0, 'tanh'
        self.weight_ih_l0 = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(hidden_size))
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size))

    def step(self, x_t, h_previous):
        """输入 [B,F]、[B,H]，返回 [B,H]；含输入与历史贡献、两份偏置、tanh。"""
        raise NotImplementedError('TODO manual.step：完成单步状态更新；不要调用现成循环层。')

    def forward(self, x, h0=None):
        """显式沿时间展开，返回 outputs[B,T,H] 与 h_n[1,B,H]。

        核对非空三维输入、F、初态尺寸/设备/dtype；默认初态跟随输入。
        时间步之间传递状态，保存每一步；不要 detach，不能每步重新清零。
        """
        raise NotImplementedError('TODO manual.forward：完成时间展开、状态传递与全部输出。')


def _check_core(model):
    if getattr(model, 'num_layers', 1) != 1 or getattr(model, 'bidirectional', False):
        raise ValueError('核对只支持单层单向。')
    if getattr(model, 'nonlinearity', 'tanh') != 'tanh' or not getattr(model, 'batch_first', True):
        raise ValueError('核对要求 tanh 与 batch_first=True。')
    if isinstance(model, (nn.LSTM, nn.GRU, nn.RNNCell)):
        raise ValueError('核心必须是普通 RNN。')
    for name in _NAMES:
        if not isinstance(getattr(model, name, None), nn.Parameter):
            raise ValueError(f'缺少参数 {name}；两份偏置必须分别保留。')


def copy_rnn_weights(source, target):
    """在同尺寸参考层和手写层之间复制四个参数；不提供递推答案。"""
    _check_core(source); _check_core(target)
    for name in _NAMES:
        if getattr(source, name).shape != getattr(target, name).shape:
            raise ValueError(f'{name} 尺寸不匹配。')
    with torch.no_grad():
        for name in _NAMES:
            getattr(target, name).copy_(getattr(source, name))
    return target


def verify_manual_rnn(student, *, batch_size=3, sequence_length=5, check_gradients=False):
    """比较相同权重、输入、非零初态下每步输出与最终状态；不更新参数/已有梯度。"""
    _check_core(student)
    if min(batch_size, sequence_length) < 1:
        raise ValueError('B/T 必须大于0。')
    # nn.RNN 只作为数值参考，不能藏进手写核心。
    if isinstance(student, (nn.RNN, nn.RNNCell)) or any(isinstance(m, (nn.RNN, nn.RNNCell, nn.LSTM, nn.GRU)) for m in student.modules() if m is not student):
        raise ValueError('手写核心不能包含现成循环层。')
    p = next(student.parameters())
    reference = nn.RNN(student.input_size, student.hidden_size, batch_first=True).to(device=p.device, dtype=p.dtype)
    copy_rnn_weights(student, reference)
    generator = torch.Generator(device='cpu').manual_seed(20261017 + batch_size + sequence_length)
    x = torch.randn(batch_size, sequence_length, student.input_size, generator=generator).to(p)
    h0 = torch.randn(1, batch_size, student.hidden_size, generator=generator).to(p)
    with evaluating(student), evaluating(reference), torch.no_grad():
        actual, last = student(x, h0)
        expected, final = reference(x, h0)
        if tuple(actual.shape) != tuple(expected.shape) or tuple(last.shape) != tuple(final.shape):
            raise AssertionError('必须返回全部时间步与[1,B,H]最终状态。')
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(last, final, rtol=1e-4, atol=1e-6)
        zero_output, zero_last = student(x)
        ref_zero, ref_last = reference(x)
        torch.testing.assert_close(zero_output, ref_zero, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(zero_last, ref_last, rtol=1e-4, atol=1e-6)
    report = {'B': batch_size, 'T': sequence_length, 'F': student.input_size, 'H': student.hidden_size,
              'output_max_error': float((actual-expected).abs().max()),
              'hidden_max_error': float((last-final).abs().max())}
    if check_gradients:
        xs, xr = x.detach().requires_grad_(), x.detach().requires_grad_()
        hs, hr = h0.detach().requires_grad_(), h0.detach().requires_grad_()
        with evaluating(student), evaluating(reference), torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            ys, ls = student(xs, hs); yr, lr = reference(xr, hr)
            ps = tuple(getattr(student, n) for n in _NAMES)
            pr = tuple(getattr(reference, n) for n in _NAMES)
            gs = torch.autograd.grad(ys.square().sum()+ls.square().sum(), (xs, hs)+ps)
            gr = torch.autograd.grad(yr.square().sum()+lr.square().sum(), (xr, hr)+pr)
            for a,b in zip(gs,gr): torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
            report['gradient_max_error'] = max(float((a-b).abs().max()) for a,b in zip(gs,gr))
    return report


def verify_manual_rnn_factory(factory, *, check_gradients=False):
    """覆盖不同 B/T/F/H；factory(F,H) 须构建使用者的核心。"""
    return [verify_manual_rnn(factory(f,h), batch_size=b, sequence_length=t, check_gradients=check_gradients)
            for b,t,f,h in ((1,1,2,3), (3,5,4,7), (2,4,8,5))]
