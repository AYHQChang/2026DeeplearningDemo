"""离线手写数字数据、随机种子和共享服务器设备选择。"""

from dataclasses import dataclass
import random

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch


@dataclass
class ImageDatasetBundle:
    """CNN 使用的统一 `[N, C, H, W]` 图像数据。"""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    class_names: tuple[str, ...]
    dataset_name: str
    pixel_range: tuple[float, float]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def channels(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def image_size(self) -> int:
        height, width = self.x_train.shape[-2:]
        if height != width:
            raise ValueError("当前小型 CNN 只接受正方形图片。")
        return int(height)


def seed_everything(seed: int) -> None:
    """固定随机性，并限制共享服务器中每个进程的 CPU 线程数。"""

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """解析 CPU、自动 CUDA 或课堂分配的 `cuda:x`。"""

    indexed_cuda = requested.startswith("cuda:") and requested[5:].isdigit()
    if requested not in {"auto", "cpu", "cuda"} and not indexed_cuda:
        raise ValueError("device 只能是 auto、cpu、cuda 或 cuda:编号。")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前环境检测不到 CUDA，请改用 cpu 或检查课程环境。")
    if indexed_cuda and int(requested[5:]) >= torch.cuda.device_count():
        raise RuntimeError(
            f"找不到第 {requested[5:]} 块 GPU；当前可见 GPU 数量为 {torch.cuda.device_count()}。"
        )
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_digits_data(seed: int = 42, test_size: float = 0.25) -> ImageDatasetBundle:
    """加载 scikit-learn 内置的 8×8 数字，并转成 CNN 四维 Tensor。"""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size 必须位于 0 和 1 之间。")
    digits = load_digits()
    images = (digits.images.astype(np.float32) / 16.0)[:, None, :, :]
    labels = digits.target.astype(np.int64)
    indices = np.arange(len(images))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    return ImageDatasetBundle(
        x_train=torch.from_numpy(images[train_indices]),
        y_train=torch.from_numpy(labels[train_indices]),
        x_test=torch.from_numpy(images[test_indices]),
        y_test=torch.from_numpy(labels[test_indices]),
        class_names=tuple(str(value) for value in range(10)),
        dataset_name="digits8",
        pixel_range=(0.0, 1.0),
    )
