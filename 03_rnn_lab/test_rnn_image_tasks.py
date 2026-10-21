"""图片作业支架回归；不包含学生转换、forward或手写递推答案。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import torch

from rnn_lab.image_tasks import (
    DATA_VERSION, find_lab_dir, prepare_fashion_archive,
    check_image_split, check_rows, check_patches,
)

LAB = Path(__file__).resolve().parent


def test_locations_and_single_notebook():
    for start in (LAB.parent, LAB, LAB / "05_image_classification"):
        assert find_lab_dir(start) == LAB
        assert (find_lab_dir(start).parent / "02_cnn_lab/data/fashion_small.zip").is_file()
    assert not (LAB / "05_自主修改练习.ipynb").exists()
    notebook = json.loads((LAB / "05_image_classification/01_服饰图片分层实验.ipynb").read_text(encoding="utf-8-sig"))
    source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    for marker in ("TODO 1", "TODO 2", "TODO 3", "TODO 4", "TODO 5", "TODO 6", "TODO 7", "TODO 8"):
        assert marker in source
    assert 'RUN_GATED = False' in source and 'RUN_MEAN = False' in source
    assert 'RUN_PATCHES = False' in source and 'RUN_MANUAL = False' in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code": compile("".join(cell["source"]), "image_notebook", "exec")
    assert DATA_VERSION == "fashion_small_original_train_val_test_3class"
    print("PASS: paths from three locations, original notebook migrated, independent switches and TODOs retained.")


def test_archive_preserves_files_and_rejects_escape():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        lab = root / "03_rnn_lab"
        data = root / "02_cnn_lab/data"
        data.mkdir(parents=True)
        existing = data / "fashion_small/train/bag/existing.png"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"user-original")
        archive_path = data / "fashion_small.zip"
        with zipfile.ZipFile(archive_path,"w") as archive:
            archive.writestr("fashion_small/train/bag/existing.png", b"replacement")
            archive.writestr("fashion_small/train/bag/new.png", b"new-file")
        assert prepare_fashion_archive(lab) == data.resolve()
        assert existing.read_bytes() == b"user-original"
        assert existing.with_name("new.png").read_bytes() == b"new-file"
        with zipfile.ZipFile(archive_path,"w") as archive:
            archive.writestr("../escape.png", b"bad")
        try: prepare_fashion_archive(lab)
        except ValueError: pass
        else: raise AssertionError("Archive traversal must be rejected.")
    print("PASS: offline extraction preserves existing files and rejects path traversal.")


def test_identity_checks():
    with TemporaryDirectory() as temporary:
        split = Path(temporary)/"val"
        paths=[]
        for name in ("x", "y"):
            (split/name).mkdir(parents=True)
            path = split/name/"one.png"
            path.write_bytes(b"identity-only")
            paths.append(path)
        images = torch.zeros(2,1,3,5)
        labels = torch.tensor([0,1],dtype=torch.int64)
        mapping={"x":0,"y":1}
        assert check_image_split(images, labels, paths, split, ("x","y"),mapping)==[1,1]
        for changed_labels,changed_paths in ((labels.flip(0),paths),(labels,list(reversed(paths)))):
            try: check_image_split(images,changed_labels,changed_paths,split,("x","y"),mapping)
            except ValueError: pass
            else: raise AssertionError("Label/path identity mismatch must be rejected.")
    print("PASS: generic classes/non-square images, label and path identity.")


def test_probe_rejects_wrong_shapes_and_fake_inverse():
    # 有意错误的替身，验证探针能抓住错误；这里没有正确转换实现。
    def wrong_rows(images): return images.flatten(1).unsqueeze(1)
    def wrong_patches(images, size):
        return images.reshape(len(images), -1, size * size)
    def fake_inverse(tokens,height,width,size):
        return tokens.reshape(len(tokens),1,height,width)
    for check in (lambda:check_rows(wrong_rows),lambda:check_patches(wrong_patches,fake_inverse)):
        try: check()
        except AssertionError: pass
        else: raise AssertionError("Probe must catch a shape mistake or mutually cancelling wrong transforms.")
    print("PASS: independent coordinates reject fake patches even when inverse reconstructs.")


def main():
    torch.set_num_threads(1)
    test_locations_and_single_notebook()
    test_archive_preserves_files_and_rejects_escape()
    test_identity_checks()
    test_probe_rejects_wrong_shapes_and_fake_inverse()
    print("Image-task scaffolding tests passed on CPU.")

if __name__ == "__main__": main()
