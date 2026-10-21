"""中文评论的离线读取、训练集词表与可核对的变长批次。"""

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import torch
from torch.utils.data import DataLoader, Dataset

PAD, UNK = "<PAD>", "<UNK>"
PAD_ID, UNK_ID = 0, 1
CLEANING_VERSION = "nfkc-whitespace-v1"


@dataclass
class TextDataBundle:
    splits: dict[str, list[dict]]
    class_names: tuple[str, ...] = ("负面", "正面")
    version: str = "unspecified"
    data_dir: Path | None = None
    manifest: dict | None = None

    @property
    def n_classes(self):
        return len(self.class_names)


def clean_text(text: str) -> str:
    """统一全半角及连续空白；保留否定、转折和标点。"""
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串。")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def validate_vocab(vocab: dict[str, int]) -> None:
    if vocab.get(PAD) != PAD_ID or vocab.get(UNK) != UNK_ID:
        raise ValueError("词表固定 <PAD>=0、<UNK>=1。")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("词表 ID 必须唯一、连续，并从 0 开始。")


def build_vocab(texts, min_frequency: int = 1, max_size: int | None = None) -> dict[str, int]:
    """只传训练集原文；频数降序、同频按字符排序，保证可复现。"""
    if min_frequency < 1 or (max_size is not None and max_size < 2):
        raise ValueError("min_frequency 至少为 1；max_size 包含两个保留符，至少为 2。")
    counts = Counter(char for text in texts for char in clean_text(text))
    chars = sorted((char for char, count in counts.items() if count >= min_frequency),
                   key=lambda char: (-counts[char], char))
    if max_size is not None:
        chars = chars[:max_size - 2]
    return {PAD: PAD_ID, UNK: UNK_ID, **{char: index + 2 for index, char in enumerate(chars)}}


def encode_text(text: str, vocab: dict[str, int], max_length: int = 100) -> dict:
    """空白文本以一个 UNK 哨兵表示，并显式标记 empty；不临时扩词表。"""
    if max_length < 1:
        raise ValueError("max_length 必须大于 0。")
    cleaned = clean_text(text)
    chars = list(cleaned[:max_length])
    ids = [vocab.get(char, UNK_ID) for char in chars]
    empty = len(ids) == 0
    if empty:
        ids = [UNK_ID]
    return {"token_ids": ids, "length": len(ids), "cleaned_text": cleaned,
            "characters": chars if chars else [UNK], "truncated": len(cleaned) > max_length,
            "unknown_count": sum(token == UNK_ID for token in ids), "empty": empty,
            "original_length": len(cleaned)}


def load_text_data(data_dir: str | Path) -> TextDataBundle:
    """读取课程离线包，不发起网络请求，不重新划分原数据。"""
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    classes = tuple(manifest.get("class_names", ("负面", "正面")))
    if len(classes) < 2:
        raise ValueError("类别配置至少包含两个类别。")
    splits = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.tsv"
        if split == "val" and not path.is_file():
            path = root / "dev.tsv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not {"id", "text", "label"}.issubset(reader.fieldnames or ()):
                raise ValueError(f"{path.name} 必须含 id、text、label 列。")
            rows = []
            for source in reader:
                label = int(source["label"])
                if not 0 <= label < len(classes) or not clean_text(source["text"]):
                    raise ValueError(f"{path.name} 含非法标签或空文本，样本 {source['id']}。")
                rows.append({"id": source["id"], "text": source["text"], "label": label})
        if not rows or len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"{path.name} 为空或样本 ID 重复。")
        expected = manifest.get("files", {}).get(split)
        if expected is not None:
            if len(rows) != expected["count"]:
                raise ValueError(f"{path.name} 实际数量与 manifest 不一致。")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
                raise ValueError(f"{path.name} SHA256 与 manifest 不一致。")
            actual_labels = {str(label): count for label, count in Counter(row["label"] for row in rows).items()}
            if actual_labels != expected["label_counts"]:
                raise ValueError(f"{path.name} 实际标签数量与 manifest 不一致。")
        splits[split] = rows
    # 课程包应在制作时处理重复，运行时再次拒绝跨划分清理后重复。
    seen = {}
    for split, rows in splits.items():
        for row in rows:
            key = clean_text(row["text"])
            if key in seen and seen[key] != split:
                raise ValueError(f"清理后文本跨 {seen[key]}/{split} 重叠，请核对数据清单。")
            seen[key] = split
    return TextDataBundle(splits, classes, str(manifest.get("version", "unspecified")), root, manifest)


class _TextDataset(Dataset):
    def __init__(self, rows, vocab, max_length):
        self.rows = rows
        self.encoded = [encode_text(row["text"], vocab, max_length) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index], self.encoded[index]


def collate_text_batch(samples) -> dict:
    """右侧补 PAD，真实长度已经过截断；T 为本批次最大有效长度。"""
    if not samples:
        raise ValueError("批次不能为空。")
    lengths = torch.tensor([encoded["length"] for _, encoded in samples], dtype=torch.int64)
    ids = torch.full((len(samples), int(lengths.max())), PAD_ID, dtype=torch.int64)
    for index, (_, encoded) in enumerate(samples):
        ids[index, :encoded["length"]] = torch.tensor(encoded["token_ids"], dtype=torch.int64)
    return {"token_ids": ids, "lengths": lengths,
            "labels": torch.tensor([row["label"] for row, _ in samples], dtype=torch.int64),
            "texts": [row["text"] for row, _ in samples], "ids": [row["id"] for row, _ in samples]}


def make_text_dataloaders(data: TextDataBundle, vocab: dict[str, int], max_length: int = 100,
                          batch_size: int = 64, seed: int = 42) -> dict[str, DataLoader]:
    validate_vocab(vocab)
    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0。")
    return {split: DataLoader(_TextDataset(rows, vocab, max_length), batch_size=batch_size,
                              shuffle=split == "train", num_workers=0, collate_fn=collate_text_batch,
                              generator=torch.Generator().manual_seed(seed))
            for split, rows in data.splits.items()}


def text_data_diagnostics(data: TextDataBundle, vocab: dict[str, int], max_length: int = 100) -> dict:
    """仅描述长度、截断与未知字；不使用标签选择模型。"""
    diagnostics = {}
    for split, rows in data.splits.items():
        encodings = [encode_text(row["text"], vocab, max_length) for row in rows]
        tokens = sum(item["length"] for item in encodings)
        diagnostics[split] = {"samples": len(rows), "tokens": tokens,
            "truncated_count": sum(item["truncated"] for item in encodings),
            "truncated_fraction": sum(item["truncated"] for item in encodings) / len(rows),
            "unknown_count": sum(item["unknown_count"] for item in encodings),
            "unknown_fraction": sum(item["unknown_count"] for item in encodings) / tokens,
            "all_unknown_count": sum(item["unknown_count"] == item["length"] for item in encodings),
            "empty_count": sum(item["empty"] for item in encodings),
            "mean_original_length": sum(item["original_length"] for item in encodings) / len(rows)}
    return diagnostics


def save_text_preprocessing(vocab, max_length, class_names, path: str | Path) -> Path:
    validate_vocab(vocab)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"vocab": vocab, "max_length": max_length, "class_names": list(class_names),
               "label_mapping": {str(index): name for index, name in enumerate(class_names)},
               "cleaning_version": CLEANING_VERSION, "padding": "right", "empty_policy": "one_UNK"}
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def load_text_preprocessing(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_vocab(payload["vocab"])
    if payload["cleaning_version"] != CLEANING_VERSION:
        raise ValueError("保存的清理规则版本与当前实现不一致。")
    return payload
