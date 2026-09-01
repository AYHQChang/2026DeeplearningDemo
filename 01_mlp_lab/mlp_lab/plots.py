"""Matplotlib figures used by notebooks and command-line examples."""


from functools import lru_cache
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from .data import DatasetBundle, make_dataset
from .engine import TrainingResult, _clone_state

POINT_COLORS = ["#2563EB", "#DC2626", "#16A34A"]
REGION_COLORS = ListedColormap(["#BFDBFE", "#FECACA", "#BBF7D0"])
BUNDLED_CJK_FONT = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "fonts"
    / "NotoSansCJKsc-Regular.otf"
)
SYSTEM_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
)


@lru_cache(maxsize=1)
def _preferred_chinese_font() -> str:
    """注册项目字体，缺失时再查找操作系统已有的中文字体。"""

    if BUNDLED_CJK_FONT.is_file():
        font_manager.fontManager.addfont(str(BUNDLED_CJK_FONT))
        return font_manager.FontProperties(fname=BUNDLED_CJK_FONT).get_name()

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in SYSTEM_CJK_FONTS:
        if candidate in available:
            return candidate

    warnings.warn(
        "未找到项目自带或系统中文字体，Matplotlib 中文可能显示为方框。请重新拉取完整项目。",
        RuntimeWarning,
        stacklevel=2,
    )
    return "DejaVu Sans"


def configure_chinese_font() -> str:
    """为 Windows 与 Linux 统一配置可显示中文的 Matplotlib 字体。"""

    selected = _preferred_chinese_font()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def _mesh(data: DatasetBundle, resolution: int=180) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    all_x = torch.cat((data.x_train, data.x_test)).numpy()
    margin = 0.55
    x_min, x_max = (all_x[:, 0].min() - margin, all_x[:, 0].max() + margin)
    y_min, y_max = (all_x[:, 1].min() - margin, all_x[:, 1].max() + margin)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32))
    return (xx, yy, grid)


def _predict_grid(result: TrainingResult, grid: torch.Tensor, state: dict[str, torch.Tensor] | None=None) -> np.ndarray:
    device = next(result.model.parameters()).device
    current_state = _clone_state(result.model)
    if state is not None:
        result.model.load_state_dict(state)
    result.model.eval()
    with torch.no_grad():
        predictions = result.model(grid.to(device)).argmax(dim=1).cpu().numpy()
    if state is not None:
        result.model.load_state_dict(current_state)
    return predictions


def _draw_boundary(axis: plt.Axes, result: TrainingResult, state: dict[str, torch.Tensor] | None=None, mark_errors: bool=False) -> None:
    xx, yy, grid = _mesh(result.data)
    zz = _predict_grid(result, grid, state=state).reshape(xx.shape)
    axis.contourf(xx, yy, zz, levels=np.arange(result.data.n_classes + 1) - 0.5, cmap=REGION_COLORS, alpha=0.72)
    train_x = result.data.x_train.numpy()
    train_y = result.data.y_train.numpy()
    axis.scatter(train_x[:, 0], train_x[:, 1], c=[POINT_COLORS[value] for value in train_y], s=13, edgecolors='white', linewidths=0.25, alpha=0.72, label='训练样本')
    if mark_errors:
        device = next(result.model.parameters()).device
        with torch.no_grad():
            predicted = result.model(result.data.x_test.to(device)).argmax(dim=1).cpu()
        wrong = predicted != result.data.y_test
        test_x = result.data.x_test.numpy()
        test_y = result.data.y_test.numpy()
        axis.scatter(test_x[:, 0], test_x[:, 1], c=[POINT_COLORS[value] for value in test_y], marker='x', s=np.where(wrong.numpy(), 50, 18), linewidths=np.where(wrong.numpy(), 1.8, 0.6), alpha=np.where(wrong.numpy(), 1.0, 0.35), label='测试样本（大叉为误判）')
    axis.set_xlabel('标准化特征 x1')
    axis.set_ylabel('标准化特征 x2')
    axis.set_aspect('equal', adjustable='box')
    axis.grid(alpha=0.12)


def plot_dataset_gallery(seed: int=42, n_samples: int=600) -> plt.Figure:
    configure_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    for axis, name, title in zip(axes, ('moons', 'circles', 'spiral'), ('月牙：局部弯曲', '同心圆：封闭边界', '螺旋：复杂非线性')):
        noise = {'moons': 0.2, 'circles': 0.2, 'spiral': 0.45}[name]
        data = make_dataset(name, n_samples=n_samples, noise=noise, seed=seed)
        points = torch.cat((data.x_train, data.x_test)).numpy()
        labels = torch.cat((data.y_train, data.y_test)).numpy()
        axis.scatter(points[:, 0], points[:, 1], c=[POINT_COLORS[value] for value in labels], s=18, alpha=0.8)
        axis.set_title(title)
        axis.set_xlabel('x1')
        axis.set_ylabel('x2')
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=0.15)
    fig.suptitle('同一个 MLP，要面对三种不同难度的数据', fontsize=15)
    return fig


def plot_training_story(result: TrainingResult) -> plt.Figure:
    configure_chinese_font()
    available_epochs = sorted(result.snapshots)
    stages = available_epochs[:2] + available_epochs[-4:]
    stages = list(dict.fromkeys(stages))[-6:]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), constrained_layout=True)
    flat_axes = axes.ravel()
    for axis, epoch in zip(flat_axes[:6], stages):
        _draw_boundary(axis, result, state=result.snapshots[epoch])
        axis.set_title(f'Epoch {epoch}：决策边界')
    epochs = np.arange(1, result.config.epochs + 1)
    curve_axis = flat_axes[6]
    curve_axis.plot(epochs, result.train_loss, color='#7C3AED', label='训练损失')
    curve_axis.set_xlabel('Epoch')
    curve_axis.set_ylabel('Loss')
    curve_axis.grid(alpha=0.2)
    accuracy_axis = curve_axis.twinx()
    accuracy_axis.plot(epochs, result.train_accuracy, color='#2563EB', label='训练准确率')
    accuracy_axis.plot(epochs, result.test_accuracy, color='#DC2626', linestyle='--', label='测试准确率')
    accuracy_axis.set_ylim(0.0, 1.02)
    accuracy_axis.set_ylabel('Accuracy')
    lines = curve_axis.lines + accuracy_axis.lines
    curve_axis.legend(lines, [line.get_label() for line in lines], loc='center right')
    curve_axis.set_title('训练是否收敛、是否过拟合？')
    info_axis = flat_axes[7]
    info_axis.axis('off')
    info_axis.text(0.02, 0.98, '\n'.join(('最终实验卡', f'数据：{result.config.dataset}', f'隐藏层：{result.config.hidden_sizes}', f'激活：{result.config.activation}', f'损失：{result.config.loss_name}', f'优化器：{result.config.optimizer}', f'Dropout：{result.config.dropout}', f'参数量：{result.parameter_count:,}', f'测试准确率：{result.final_test_accuracy:.1%}', f'训练耗时：{result.elapsed_seconds:.2f} 秒')), va='top', fontsize=12, linespacing=1.55)
    fig.suptitle(f'训练全过程：{result.config.name}', fontsize=16)
    return fig


def plot_final_diagnosis(result: TrainingResult) -> plt.Figure:
    configure_chinese_font()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    _draw_boundary(axes[0], result, mark_errors=True)
    axes[0].set_title(f'最终边界｜测试 {result.final_test_accuracy:.1%}')
    axes[0].legend(fontsize=8, loc='upper right')
    epochs = np.arange(1, result.config.epochs + 1)
    axes[1].plot(epochs, result.train_loss, color='#7C3AED')
    axes[1].set_title('Loss：下降是否稳定？')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[2].plot(epochs, result.train_accuracy, color='#2563EB', label='训练')
    axes[2].plot(epochs, result.test_accuracy, color='#DC2626', linestyle='--', label='测试')
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title('Accuracy：泛化差距多大？')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[3].plot(epochs, result.gradient_norm, color='#16A34A')
    axes[3].set_yscale('log')
    axes[3].set_title('Gradient norm：更新信号多强？')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('平均 L2 norm（对数轴）')
    for axis in axes[1:]:
        axis.grid(alpha=0.2)
    fig.suptitle(result.config.name, fontsize=15)
    return fig


def plot_comparison(results: list[TrainingResult], title: str) -> plt.Figure:
    if not results:
        raise ValueError('results 不能为空。')
    configure_chinese_font()
    n = len(results)
    fig, axes = plt.subplots(4, n, figsize=(5.0 * n, 15.2), squeeze=False, constrained_layout=True)
    for column, result in enumerate(results):
        _draw_boundary(axes[0, column], result, mark_errors=True)
        axes[0, column].set_title(f'{result.config.name}\n测试 {result.final_test_accuracy:.1%}｜参数 {result.parameter_count:,}｜{result.elapsed_seconds:.2f}s')
        epochs = np.arange(1, result.config.epochs + 1)
        axes[1, column].plot(epochs, result.train_loss, color='#7C3AED')
        axes[1, column].set_title('训练损失')
        axes[1, column].set_xlabel('Epoch')
        axes[1, column].set_ylabel('Loss')
        axes[2, column].plot(epochs, result.train_accuracy, color='#2563EB', label='训练')
        axes[2, column].plot(epochs, result.test_accuracy, color='#DC2626', linestyle='--', label='测试')
        axes[2, column].set_ylim(0.0, 1.02)
        axes[2, column].set_title('训练/测试准确率')
        axes[2, column].set_xlabel('Epoch')
        axes[2, column].set_ylabel('Accuracy')
        axes[2, column].legend()
        axes[3, column].plot(epochs, result.gradient_norm, color='#16A34A')
        axes[3, column].set_yscale('log')
        axes[3, column].set_title('平均梯度范数（对数轴）')
        axes[3, column].set_xlabel('Epoch')
        axes[3, column].set_ylabel('L2 norm')
        for row in (1, 2, 3):
            axes[row, column].grid(alpha=0.2)
    fig.suptitle(title + '｜每一列只替换一个目标组件', fontsize=17)
    return fig


def format_result_table(results: list[TrainingResult]) -> str:
    lines = ['配置对比', '-' * 86, f"{'配置':<30}{'测试准确率':>12}{'参数量':>12}{'耗时(秒)':>12}{'末轮梯度':>14}", '-' * 86]
    for result in results:
        lines.append(f'{result.config.name:<30}{result.final_test_accuracy:>11.1%}{result.parameter_count:>12,}{result.elapsed_seconds:>12.2f}{result.gradient_norm[-1]:>14.4g}')
    lines.append('-' * 86)
    return '\n'.join(lines)


def print_result_table(results: list[TrainingResult]) -> None:
    print('\n' + format_result_table(results))


def save_or_show(figure: plt.Figure, filename: str, save_dir: str | Path | None=None, show: bool=True) -> None:
    if save_dir is not None:
        destination = Path(save_dir)
        destination.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination / filename, dpi=150, bbox_inches='tight')
        print(f'已保存图像：{destination / filename}')
    if show:
        plt.show()
    else:
        plt.close(figure)
