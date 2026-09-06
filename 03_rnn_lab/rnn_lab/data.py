"""延迟回忆任务的数据生成、随机种子和设备选择。"""

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = ("红色", "绿色", "蓝色")


@dataclass
class SequenceDatasetBundle:
    """RNN 使用的固定长度序列数据，输入统一为 ``[N, T, F]``。"""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    class_names: tuple[str, ...] = CLASS_NAMES

    @property
    def sequence_length(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def input_size(self) -> int:
        return int(self.x_train.shape[2])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def seed_everything(seed: int) -> None:
    """固定随机性，并限制共享服务器中每个进程的 CPU 线程数。"""

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """解析 CPU、CUDA 或课堂分配的 ``cuda:x``，并检查编号。"""

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


def _balanced_labels(size: int, rng: np.random.Generator) -> np.ndarray:
    """生成数量尽可能相等的三类标签，避免类别不平衡干扰结论。"""

    labels = np.arange(size, dtype=np.int64) % len(CLASS_NAMES)
    rng.shuffle(labels)
    return labels


def _make_split(
    size: int,
    sequence_length: int,
    rng: np.random.Generator,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成一个数据划分：首步写秘密，中间写噪声，末步写查询标记。"""

    labels = _balanced_labels(size, rng)
    # 6 个特征通道：3 个秘密类别 + 2 个随机干扰 + 1 个查询标记。
    sequences = np.zeros((size, sequence_length, 6), dtype=np.float32)
    sequences[np.arange(size), 0, labels] = 1.0
    if sequence_length > 2:
        sequences[:, 1:-1, 3:5] = rng.normal(
            0.0, noise_scale, size=(size, sequence_length - 2, 2)
        ).astype(np.float32)
    sequences[:, -1, 5] = 1.0
    return torch.from_numpy(sequences), torch.from_numpy(labels)


def make_delayed_recall_data(
    sequence_length: int = 30,
    train_size: int = 900,
    val_size: int = 240,
    test_size: int = 240,
    seed: int = 42,
    noise_scale: float = 0.5,
) -> SequenceDatasetBundle:
    """生成完全离线的三分类延迟回忆数据。"""

    if sequence_length < 3:
        raise ValueError("sequence_length 至少为 3：秘密、干扰和查询各占一步。")
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("train_size、val_size 和 test_size 必须大于 0。")
    if noise_scale < 0:
        raise ValueError("noise_scale 不能小于 0。")

    rng = np.random.default_rng(seed)
    x_train, y_train = _make_split(train_size, sequence_length, rng, noise_scale)
    x_val, y_val = _make_split(val_size, sequence_length, rng, noise_scale)
    x_test, y_test = _make_split(test_size, sequence_length, rng, noise_scale)
    return SequenceDatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


def make_dataloaders(
    data: SequenceDatasetBundle,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """创建训练、验证和测试 DataLoader；不启动额外工作进程。"""

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")
    datasets = {
        "train": TensorDataset(data.x_train, data.y_train),
        "val": TensorDataset(data.x_val, data.y_val),
        "test": TensorDataset(data.x_test, data.y_test),
    }
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )
        for name, dataset in datasets.items()
    }
