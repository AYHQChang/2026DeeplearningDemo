"""Offline datasets and device selection shared by all MLP examples."""


from dataclasses import dataclass
import random

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


@dataclass
class DatasetBundle:
    x_train_raw: np.ndarray
    x_test_raw: np.ndarray
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    n_classes: int
    dataset_name: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray


def seed_everything(seed: int) -> None:
    # 共享服务器课堂模式：避免几十个 Notebook 各自占满全部 CPU 线程。
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    indexed_cuda = requested.startswith('cuda:') and requested[5:].isdigit()
    if requested not in {'auto', 'cpu', 'cuda'} and not indexed_cuda:
        raise ValueError('device 只能是 auto、cpu、cuda 或 cuda:编号。')
    if requested.startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('当前 PyTorch 环境检测不到 CUDA。请改用 --device cpu，或课前安装与显卡驱动匹配的 CUDA 版 PyTorch。')
    if indexed_cuda and int(requested[5:]) >= torch.cuda.device_count():
        raise RuntimeError(f'找不到第 {requested[5:]} 块 GPU；当前可见 GPU 数量为 {torch.cuda.device_count()}。')
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)


def _make_spiral(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_classes = 3
    per_class = n_samples // n_classes
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for class_id in range(n_classes):
        radius = np.linspace(0.05, 1.0, per_class)
        angle = np.linspace(class_id * 4.0, (class_id + 1) * 4.0, per_class)
        angle += rng.normal(0.0, noise, per_class)
        points = np.column_stack((radius * np.sin(angle), radius * np.cos(angle)))
        xs.append(points)
        ys.append(np.full(per_class, class_id, dtype=np.int64))
    return (np.vstack(xs).astype(np.float32), np.concatenate(ys))


def make_dataset(name: str, n_samples: int=600, noise: float=0.2, seed: int=42) -> DatasetBundle:
    if name == 'moons':
        x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif name == 'circles':
        x, y = make_circles(n_samples=n_samples, noise=noise * 0.55, factor=0.45, random_state=seed)
    elif name == 'spiral':
        x, y = _make_spiral(n_samples=n_samples, noise=noise, seed=seed)
    else:
        raise ValueError('dataset 只能是 moons、circles 或 spiral。')
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    x_train_raw = x_train_raw.astype(np.float32)
    x_test_raw = x_test_raw.astype(np.float32)
    scaler = StandardScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)
    return DatasetBundle(x_train_raw=x_train_raw, x_test_raw=x_test_raw, x_train=torch.from_numpy(x_train), y_train=torch.from_numpy(y_train).long(), x_test=torch.from_numpy(x_test), y_test=torch.from_numpy(y_test).long(), n_classes=int(np.max(y)) + 1, dataset_name=name, feature_mean=scaler.mean_.copy(), feature_scale=scaler.scale_.copy())
