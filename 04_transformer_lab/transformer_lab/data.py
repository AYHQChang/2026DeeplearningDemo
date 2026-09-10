"""注意力寻宝任务的数据生成、设备选择和 DataLoader。"""

from dataclasses import dataclass
from math import ceil, factorial
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = ("红色", "绿色", "蓝色")
TOKEN_NAMES = ("红", "绿", "蓝", "查询")
QUERY_ID = 3


@dataclass
class RetrievalDatasetBundle:
    """固定长度 token 数据，输入统一为 ``[N, T]``。"""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    class_names: tuple[str, ...] = CLASS_NAMES
    token_names: tuple[str, ...] = TOKEN_NAMES

    @property
    def input_length(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def content_length(self) -> int:
        return self.input_length - 1

    @property
    def vocab_size(self) -> int:
        return len(self.token_names)


def seed_everything(seed: int) -> None:
    """固定随机性，并限制每个课堂进程只使用一个 CPU 线程。"""

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """只接受 CPU 或教师明确分配的 ``cuda:x``。"""

    indexed_cuda = requested.startswith("cuda:") and requested[5:].isdigit()
    if requested != "cpu" and not indexed_cuda:
        raise ValueError('device 只能是 "cpu" 或课堂分配的 "cuda:编号"；不允许 auto。')
    if indexed_cuda and not torch.cuda.is_available():
        raise RuntimeError("当前环境检测不到 CUDA，请改用 cpu 或检查课程环境。")
    if indexed_cuda and int(requested[5:]) >= torch.cuda.device_count():
        raise RuntimeError(
            f"找不到第 {requested[5:]} 块 GPU；当前可见 GPU 数量为 {torch.cuda.device_count()}。"
        )
    return torch.device(requested)


def _capacity_per_label(content_length: int) -> int:
    each = content_length // len(CLASS_NAMES)
    return factorial(content_length - 1) // (
        factorial(each - 1) * factorial(each) * factorial(each)
    )


def _balanced_labels(size: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.arange(size, dtype=np.int64) % len(CLASS_NAMES)
    rng.shuffle(labels)
    return labels


def _make_split(
    size: int,
    content_length: int,
    seed: int,
    used_sequences: set[bytes],
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成一个类别平衡且不与其他划分重复的数据集。"""

    rng = np.random.default_rng(seed)
    each = content_length // len(CLASS_NAMES)
    labels = _balanced_labels(size, rng)
    rows: list[np.ndarray] = []
    for label in labels:
        counts = [each, each, each]
        counts[int(label)] -= 1
        remaining = np.repeat(np.arange(3, dtype=np.uint8), counts)
        while True:
            rng.shuffle(remaining)
            content = np.concatenate((np.array([label], dtype=np.uint8), remaining))
            key = content.tobytes()
            if key not in used_sequences:
                used_sequences.add(key)
                rows.append(np.append(content, QUERY_ID))
                break
    return torch.from_numpy(np.stack(rows).astype(np.int64)), torch.from_numpy(labels)


def make_retrieval_data(
    content_length: int = 12,
    train_size: int = 1200,
    val_size: int = 300,
    test_size: int = 300,
    seed: int = 42,
) -> RetrievalDatasetBundle:
    """离线生成“首位颜色 + 平衡干扰颜色 + QUERY”的三分类数据。"""

    if content_length < 3 or content_length % len(CLASS_NAMES) != 0:
        raise ValueError("content_length 至少为 3，且必须能被 3 整除。")
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("train_size、val_size 和 test_size 必须大于 0。")
    capacity = _capacity_per_label(content_length)
    if sum(ceil(size / 3) for size in (train_size, val_size, test_size)) > capacity:
        raise ValueError("当前长度可生成的无重复序列不足，请增大 content_length 或减小数据量。")

    used_sequences: set[bytes] = set()
    splits = [
        _make_split(size, content_length, seed + offset, used_sequences)
        for size, offset in ((train_size, 0), (val_size, 1), (test_size, 2))
    ]
    return RetrievalDatasetBundle(
        x_train=splits[0][0], y_train=splits[0][1],
        x_val=splits[1][0], y_val=splits[1][1],
        x_test=splits[2][0], y_test=splits[2][1],
    )


def make_dataloaders(
    data: RetrievalDatasetBundle,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """创建三个 DataLoader；共享服务器不启动额外工作进程。"""

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
