"""测试 dataset.py：类别权重除零/缺失类防护、图像预处理与 HAMDataset。"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset import (
    CLASSES,
    HAMDataset,
    center_crop_resize,
    compute_class_sample_weights,
    load_bgr,
    read_splits,
)


def test_compute_class_sample_weights_normal_imbalanced() -> None:
    """不平衡分布下最小类权重最大，每类总采样期望相等。"""
    labels = [0] * 2 + [1] * 10 + [5] * 100
    counts, class_w, sample_w = compute_class_sample_weights(labels, num_classes=7)

    assert counts.tolist() == [2, 10, 0, 0, 0, 100, 0]
    assert np.all(np.isfinite(class_w))
    assert np.all(np.isfinite(sample_w))
    assert class_w[0] == pytest.approx(0.5)
    assert class_w[1] == pytest.approx(0.1)
    assert class_w[5] == pytest.approx(0.01)
    # 缺失类别安全置 0.0，绝不产生 inf 或 NaN
    for missing_cls in (2, 3, 4, 6):
        assert class_w[missing_cls] == 0.0
    # 所有已出现类别的总采样质量相等（均为 1.0）
    for cls in (0, 1, 5):
        total_mass = sample_w[np.array(labels) == cls].sum()
        assert total_mass == pytest.approx(1.0)


def test_compute_class_sample_weights_zero_count_no_warning() -> None:
    """存在缺失类别时绝不触发 RuntimeWarning（divide by zero）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        counts, class_w, sample_w = compute_class_sample_weights([5, 5, 5], num_classes=7)
    assert counts[5] == 3
    assert class_w[0] == 0.0
    assert sample_w.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_compute_class_sample_weights_invalid_inputs() -> None:
    """空标签、非法类别数与越界标签必须抛 ValueError。"""
    with pytest.raises(ValueError, match="为空"):
        compute_class_sample_weights([], num_classes=7)
    with pytest.raises(ValueError, match="正整数"):
        compute_class_sample_weights([0], num_classes=0)
    with pytest.raises(ValueError, match="越界"):
        compute_class_sample_weights([-1, 0], num_classes=7)
    with pytest.raises(ValueError, match="越界"):
        compute_class_sample_weights([0, 7], num_classes=7)


def test_center_crop_resize_and_load_bgr_unicode_path(tmp_path: Path) -> None:
    """验证非方形图中心裁剪缩放，以及中文路径读写与缺失文件异常。"""
    img = np.zeros((100, 160, 3), dtype=np.uint8)
    img[:, 30:130] = (10, 120, 220)  # 中心 100×100 区域着色
    resized = center_crop_resize(img, size=64)
    assert resized.shape == (64, 64, 3)
    assert np.all(resized == np.array([10, 120, 220], dtype=np.uint8))

    with pytest.raises(ValueError):
        center_crop_resize(img, size=0)

    cn_dir = tmp_path / "皮肤镜样例"
    cn_dir.mkdir()
    img_path = cn_dir / "ISIC_测试.jpg"
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    img_path.write_bytes(encoded.tobytes())

    loaded = load_bgr(img_path)
    assert loaded.shape == (100, 160, 3)

    with pytest.raises(OSError):
        load_bgr(cn_dir / "not_exists.jpg")


def test_read_splits_and_dataset_init_validation(tmp_path: Path) -> None:
    """验证 UTF-8 BOM CSV 读取与 HAMDataset 初始化参数校验（无需 PyTorch）。"""
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text(
        "image_id,lesion_id,dx,label,split\n"
        "ISIC_0001,HAM_0001,mel,4,train\n"
        "ISIC_0002,HAM_0002,nv,5,val\n",
        encoding="utf-8-sig",
    )
    rows = read_splits(splits_csv)
    assert len(rows) == 2
    assert rows[0]["image_id"] == "ISIC_0001"

    ds = HAMDataset(tmp_path / "images", splits_csv, split="train", augment="base", size=64)
    assert len(ds) == 1
    assert ds.labels == [4]
    assert ds.keys == ["ISIC_0001"]

    with pytest.raises(ValueError, match="未知增强"):
        HAMDataset(tmp_path / "images", splits_csv, split="train", augment="invalid")
    with pytest.raises(ValueError, match="未找到 split"):
        HAMDataset(tmp_path / "images", splits_csv, split="test")


def test_ham_dataset_getitem_with_torch(tmp_path: Path) -> None:
    """有 PyTorch 环境时验证张量输出形状、归一化数值与强增强运行。"""
    torch = pytest.importorskip("torch")

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    raw = np.full((80, 120, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", raw)
    assert ok
    (img_dir / "ISIC_0001.jpg").write_bytes(encoded.tobytes())

    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text(
        f"image_id,lesion_id,dx,label,split\nISIC_0001,HAM_0001,{CLASSES[4]},4,train\n",
        encoding="utf-8",
    )

    for aug in ("none", "base", "strong"):
        ds = HAMDataset(img_dir, splits_csv, split="train", augment=aug, size=64)
        tensor, label, key = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tuple(tensor.shape) == (3, 64, 64)
        assert torch.isfinite(tensor).all()
        assert label == 4
        assert key == "ISIC_0001"
