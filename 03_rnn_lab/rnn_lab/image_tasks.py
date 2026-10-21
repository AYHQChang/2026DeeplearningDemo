"""图片作业的路径、离线包和独立探针；不包含学生转换或模型答案。"""

from pathlib import Path
import zipfile

import torch


DATA_VERSION = "fashion_small_original_train_val_test_3class"


def find_lab_dir(start=None):
    """从仓库、实验室、任务目录（及其子目录）定位实验室。"""
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()
    if start.is_file():
        start = start.parent
    for directory in (start, *start.parents):
        for candidate in (directory, directory / "03_rnn_lab"):
            if (candidate / "rnn_lab").is_dir() and (candidate / "README.md").is_file():
                return candidate
    raise FileNotFoundError("找不到 03_rnn_lab；请从课程仓库或对应任务目录运行。")


def prepare_fashion_archive(lab_dir):
    """仅解压已有离线包；原图片与已有文件均不覆盖，不联网。"""
    archive_path = Path(lab_dir).parent / "02_cnn_lab" / "data" / "fashion_small.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(f"缺少课程离线包：{archive_path}。请使用完整课程仓库。")
    destination_root = archive_path.parent.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (destination_root / member.filename).resolve()
            if not destination.is_relative_to(destination_root):
                raise ValueError("离线图片包包含越界路径，停止解压。")
            if not destination.exists():
                archive.extract(member, destination_root)
    return destination_root


def check_image_split(images, labels, paths, split_dir, class_names, mapping):
    """核对像素、标签、路径身份和排序，防止样本丢失或混入其他划分。"""
    split_dir = Path(split_dir)
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("图片必须是 torch.Tensor [N,1,H,W]。")
    if min(images.shape) < 1 or images.dtype != torch.float32 or not bool(torch.isfinite(images).all()):
        raise ValueError("图片必须是非空、有限的 float32 张量。")
    if float(images.min()) < 0 or float(images.max()) > 1:
        raise ValueError("图片像素应在 [0,1]。")
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64:
        raise ValueError("标签必须是 torch.int64。")
    if tuple(labels.shape) != (len(images),) or len(paths) != len(images):
        raise ValueError("图片、标签、路径必须一一对应。")
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("至少需要两个互不重复的类别。")
    if mapping != {name: index for index, name in enumerate(class_names)}:
        raise ValueError("类别映射必须与 class_names 顺序一致且从0连续编号。")
    actual_classes = {p.name for p in split_dir.iterdir() if p.is_dir()}
    if actual_classes != set(class_names):
        raise ValueError("划分目录中的类别与配置不一致。")
    expected_paths = [p for name in class_names for p in sorted((split_dir / name).glob("*.png"))]
    if not expected_paths or [Path(p).resolve() for p in paths] != [p.resolve() for p in expected_paths]:
        raise ValueError("路径丢失、顺序错误或混入了其他划分。")
    expected_labels = torch.tensor([mapping[p.parent.name] for p in expected_paths], dtype=torch.int64)
    if not torch.equal(labels.cpu(), expected_labels):
        raise ValueError("标签与路径类别不一致。")
    return torch.bincount(labels, minlength=len(class_names)).tolist()


def _expect_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("非法输入必须给出清楚的 ValueError。")


def check_rows(converter):
    """非方形、多样本独立坐标检查；检查顺序而不只检查 shape。"""
    for batch, height, width in ((2, 3, 5), (3, 6, 8), (1, 1, 7)):
        images = torch.arange(batch * height * width, dtype=torch.float32).reshape(batch, 1, height, width)
        before = images.clone()
        result = converter(images)
        assert tuple(result.shape) == (batch, height, width), "逐行读取：T=H、F=W。"
        assert result.dtype == images.dtype, "转换应保留像素类型。"
        for sample in range(batch):
            for row in range(height):
                torch.testing.assert_close(result[sample, row], images[sample, 0, row], rtol=0, atol=0)
        torch.testing.assert_close(images, before, rtol=0, atol=0)
    for invalid in (torch.zeros(2, 3, 5), torch.zeros(2, 2, 3, 5), torch.empty(0, 1, 3, 5)):
        _expect_value_error(lambda invalid=invalid: converter(invalid))
    return "逐行：多样本、非方形、坐标与非法输入检查通过。"


def check_patches(converter, inverse):
    """独立正向空间坐标、逆变换与非法参数检查；不提供转换实现。"""
    for batch, height, width, size in ((2, 28, 28, 4), (3, 12, 20, 4), (2, 6, 8, 2), (1, 3, 5, 1)):
        images = torch.arange(batch * height * width, dtype=torch.float32).reshape(batch, 1, height, width)
        before = images.clone()
        tokens = converter(images, size)
        assert tuple(tokens.shape) == (batch, (height // size) * (width // size), size * size)
        assert tokens.dtype == images.dtype
        # 直接按原图坐标读取真值，独立于学生正逆变换。
        for sample in range(batch):
            for block_row in range(height // size):
                for block_col in range(width // size):
                    step = block_row * (width // size) + block_col
                    for inner_row in range(size):
                        for inner_col in range(size):
                            expected = images[sample, 0, block_row * size + inner_row, block_col * size + inner_col]
                            assert tokens[sample, step, inner_row * size + inner_col] == expected, "块行优先或块内行优先坐标错误。"
        torch.testing.assert_close(inverse(tokens, height, width, size), images, rtol=0, atol=0)
        torch.testing.assert_close(images, before, rtol=0, atol=0)
    valid = torch.zeros(2, 1, 12, 20)
    for invalid_size in (0, -1, 2.5, True):
        _expect_value_error(lambda invalid_size=invalid_size: converter(valid, invalid_size))
        _expect_value_error(lambda invalid_size=invalid_size: inverse(torch.zeros(2, 15, 16), 12, 20, invalid_size))
    for invalid in (torch.zeros(2, 12, 20), torch.zeros(2, 2, 12, 20), torch.zeros(2, 1, 13, 20), torch.empty(0, 1, 12, 20)):
        _expect_value_error(lambda invalid=invalid: converter(invalid, 4))
    for tokens, height, width in ((torch.zeros(2, 14, 16), 12, 20), (torch.zeros(2, 15, 15), 12, 20), (torch.zeros(2, 15, 16), 13, 20), (torch.zeros(15, 16), 12, 20)):
        _expect_value_error(lambda tokens=tokens, height=height, width=width: inverse(tokens, height, width, 4))
    return "分块：独立空间坐标、多样本、非方形、逆变换及非法参数检查通过。"
