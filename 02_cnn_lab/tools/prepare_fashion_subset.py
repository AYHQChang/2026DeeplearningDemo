"""教师用：从 Fashion-MNIST 官方 IDX 文件制作离线三类 PNG 练习包。"""

from __future__ import annotations

import gzip
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "fashion_small.zip"
SOURCE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
# 官方标签编号；目录名按字母排序后与 Notebook 的 0、1、2 映射一致。
CLASSES = {"bag": 8, "sneaker": 7, "trousers": 1}
COUNTS = {"train": 300, "val": 100, "test": 100}
SEED = 42
EXPECTED_SOURCE_SHA256 = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}


def download(name: str, folder: Path) -> tuple[bytes, str]:
    path = folder / name
    with urlopen(SOURCE + name, timeout=60) as response:
        compressed = response.read()
    actual_hash = sha256(compressed).hexdigest()
    if actual_hash != EXPECTED_SOURCE_SHA256[name]:
        raise ValueError(f"源文件 {name} 的 SHA256 不匹配；停止生成数据包")
    path.write_bytes(compressed)
    return gzip.decompress(compressed), actual_hash


def read_images(raw: bytes) -> np.ndarray:
    magic, count, height, width = struct.unpack(">IIII", raw[:16])
    if magic != 2051 or (height, width) != (28, 28):
        raise ValueError("Fashion-MNIST 图片头或尺寸错误")
    images = np.frombuffer(raw, dtype=np.uint8, offset=16)
    if images.size != count * height * width:
        raise ValueError("Fashion-MNIST 图片数量与文件长度不一致")
    return images.reshape(count, height, width)


def read_labels(raw: bytes) -> np.ndarray:
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError("Fashion-MNIST 标签文件头错误")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    if labels.size != count:
        raise ValueError("Fashion-MNIST 标签数量与文件长度不一致")
    return labels


def main() -> None:
    with TemporaryDirectory(prefix="fashion_mnist_source_") as temporary:
        folder = Path(temporary)
        contents = {}
        hashes = {}
        for key, filename in FILES.items():
            contents[key], hashes[filename] = download(filename, folder)
        images = {"train": read_images(contents["train_images"]),
                  "test": read_images(contents["test_images"])}
        labels = {"train": read_labels(contents["train_labels"]),
                  "test": read_labels(contents["test_labels"])}
        if len(images["train"]) != len(labels["train"]) or len(images["test"]) != len(labels["test"]):
            raise ValueError("图片与标签数量不一致")
        rng = np.random.default_rng(SEED)
        selection: dict[str, dict[str, list[int]]] = {split: {} for split in COUNTS}
        for name, label in CLASSES.items():
            train_indices = rng.permutation(np.flatnonzero(labels["train"] == label))
            test_indices = rng.permutation(np.flatnonzero(labels["test"] == label))
            selection["train"][name] = train_indices[:COUNTS["train"]].tolist()
            selection["val"][name] = train_indices[COUNTS["train"]:COUNTS["train"] + COUNTS["val"]].tolist()
            selection["test"][name] = test_indices[:COUNTS["test"]].tolist()
        manifest = {
            "source": "Zalando Research Fashion-MNIST",
            "source_url": "https://github.com/zalandoresearch/fashion-mnist",
            "source_license": "MIT; see official repository LICENSE",
            "source_file_sha256": hashes,
            "seed": SEED,
            "class_to_original_label": CLASSES,
            "class_to_new_label": {name: i for i, name in enumerate(sorted(CLASSES))},
            "count_per_class": COUNTS,
            "train_val_from_original_train": True,
            "test_from_original_test": True,
            "image_format": "28x28 8-bit grayscale PNG",
            "selection_original_indices": selection,
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for split in ("train", "val", "test"):
                original_split = "train" if split != "test" else "test"
                for name in sorted(CLASSES):
                    for index in sorted(selection[split][name]):
                        image = Image.fromarray(images[original_split][index], mode="L")
                        buffer = BytesIO()
                        image.save(buffer, format="PNG", optimize=True)
                        entry = ZipInfo(f"fashion_small/{split}/{name}/{index:05d}.png", date_time=(2026, 9, 23, 0, 0, 0))
                        entry.compress_type = ZIP_DEFLATED
                        archive.writestr(entry, buffer.getvalue(), compress_type=ZIP_DEFLATED, compresslevel=9)
            entry = ZipInfo("fashion_small/MANIFEST.json", date_time=(2026, 9, 23, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            archive.writestr(entry, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    license_path = OUTPUT.parent / "FASHION_MNIST_LICENSE.txt"
    with urlopen("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE", timeout=30) as response:
        license_bytes = response.read()
    if sha256(license_bytes).hexdigest() != "13ef4788476d292858fa60eb9a5f74aeca5c65770bc885ccaa05823a17ef7be1":
        raise ValueError("Fashion-MNIST 官方 LICENSE 校验失败")
    license_path.write_bytes(license_bytes)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
    print(f"SHA256 {sha256(OUTPUT.read_bytes()).hexdigest()}")
    print(f"Images: {sum(COUNTS.values()) * len(CLASSES)}")


if __name__ == "__main__":
    main()
