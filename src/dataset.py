"""HAM10000 数据集：读取 prepare.py 生成的 splits.csv 与图像。

基线与改进共用同一数据接口，差别只在增强强度（strong 增加颜色扰动、模糊与小角度旋转）。
同时提供不依赖 PyTorch 的类别采样权重计算、安全读图与图像裁剪预处理工具。
"""

from __future__ import annotations

import csv
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # 允许在无 PyTorch 环境中安全复用数据读写与权重工具
    torch = None  # type: ignore[assignment]
    _TorchDataset = object  # type: ignore[misc,assignment]

CLASSES: list[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
VALID_AUGMENTS: tuple[str, ...] = ("none", "base", "strong")
MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet 均值
STD: tuple[float, float, float] = (0.229, 0.224, 0.225)  # ImageNet 标准差


def load_bgr(path: str | Path) -> np.ndarray:
    """读取 BGR 图像（兼容中文路径与 Windows 环境）。

    Args:
        path: 图像文件路径。

    Returns:
        ``(H, W, 3)`` 的 ``uint8`` BGR 图像数组。

    Raises:
        OSError: 文件不存在或解码失败。
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise OSError(f"图像读取失败（文件不存在）: {file_path}")
    raw = np.fromfile(str(file_path), dtype=np.uint8)
    if raw.size == 0:
        raise OSError(f"图像读取失败（空文件）: {file_path}")
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"图像读取失败: {file_path}")
    return img


def read_splits(csv_path: str | Path) -> list[dict[str, str]]:
    """读取 ``splits.csv`` 并返回记录列表（兼容 UTF-8 BOM）。"""
    with open(csv_path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def center_crop_resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    """对图像执行中心方形裁剪并缩放到 ``size × size``。

    Args:
        img: 输入图像数组（至少 2 维，且宽高均大于 0）。
        size: 输出边长像素。

    Returns:
        缩放后的方形图像数组。

    Raises:
        ValueError: ``size <= 0`` 或输入图像尺寸非法。
    """
    if size <= 0:
        raise ValueError(f"目标尺寸 size 必须为正整数，当前为: {size}")
    if img.ndim < 2 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError(f"非法输入图像形状: {img.shape}")
    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = img[top : top + side, left : left + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def compute_class_sample_weights(
    labels: Sequence[int],
    num_classes: int = len(CLASSES),
    min_weight: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按类别频率倒数安全计算每类与每样本采样权重（防护除零与缺失类）。

    对于样本数为 0 的缺失类别，其类权重安全置为 ``0.0``，绝不产生 ``inf`` 或 ``NaN``。

    Args:
        labels: 样本整型标签序列，取值范围 ``[0, num_classes - 1]``。
        num_classes: 类别总数。
        min_weight: 已出现类别的最小权重下界。

    Returns:
        三元组 ``(counts, class_weights, sample_weights)``：
        - ``counts``: 形状 ``(num_classes,)`` 的各类样本数数组；
        - ``class_weights``: 形状 ``(num_classes,)`` 的各类采样权重（缺失类为 0.0）；
        - ``sample_weights``: 形状 ``(len(labels),)`` 的逐样本采样权重。

    Raises:
        ValueError: ``num_classes <= 0``、``labels`` 为空或包含越界标签。
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes 必须为正整数，当前为: {num_classes}")
    if len(labels) == 0:
        raise ValueError("labels 为空，无法计算类别采样权重")

    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.ndim != 1:
        raise ValueError(f"labels 必须为一维序列，当前维度为: {labels_arr.ndim}")
    if np.any(labels_arr < 0) or np.any(labels_arr >= num_classes):
        raise ValueError(
            f"labels 包含越界类别索引（合法范围 0..{num_classes - 1}）: "
            f"min={int(labels_arr.min())}, max={int(labels_arr.max())}"
        )

    counts = np.bincount(labels_arr, minlength=num_classes)
    class_weights = np.zeros(num_classes, dtype=np.float64)
    nonzero = counts > 0
    class_weights[nonzero] = np.clip(
        1.0 / counts[nonzero].astype(np.float64), a_min=min_weight, a_max=None
    )
    sample_weights = class_weights[labels_arr]
    return counts, class_weights, sample_weights


class HAMDataset(_TorchDataset):
    """HAM10000 皮肤镜图像数据集。

    Args:
        img_dir: 图像目录路径。
        splits_csv: ``prepare.py`` 或 ``parquet_extract.py`` 生成的 ``splits.csv`` 路径。
        split: 数据集划分名称（``train`` / ``val`` / ``test``）。
        augment: 增强档位（``none`` / ``base`` / ``strong``）。
        size: 输出方形输入分辨率，默认 ``224``。
    """

    def __init__(
        self,
        img_dir: str | Path,
        splits_csv: str | Path,
        split: str,
        augment: str = "none",
        size: int = 224,
    ) -> None:
        if augment not in VALID_AUGMENTS:
            raise ValueError(f"未知增强模式 {augment!r}，可选: {VALID_AUGMENTS}")
        if size <= 0:
            raise ValueError(f"size 必须为正整数，当前为: {size}")

        self.dir = Path(img_dir)
        rows = [r for r in read_splits(splits_csv) if r.get("split") == split]
        if not rows:
            raise ValueError(f"在 {splits_csv} 中未找到 split={split!r} 的任何样本")

        self.files: list[Path] = [self.dir / f"{r['image_id']}.jpg" for r in rows]
        self.labels: list[int] = [int(r["label"]) for r in rows]
        self.keys: list[str] = [r["image_id"] for r in rows]
        self.augment = augment
        self.size = size

    def __len__(self) -> int:
        return len(self.files)

    def _aug(self, img: np.ndarray, label: int) -> np.ndarray:
        """对单张 RGB 图像执行在线数据增强。"""
        del label  # 保留参数签名兼容
        rng = random
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.5:
            img = img[::-1]
        k = rng.randint(0, 3)
        img = np.rot90(img, k)
        if self.augment == "strong":
            if rng.random() < 0.8:  # 皮肤镜图像色温与亮度差异大，做适度亮度/对比度扰动
                img = cv2.convertScaleAbs(
                    img, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-20, 20)
                )
            if rng.random() < 0.5:
                img = cv2.GaussianBlur(img, (3, 3), 0)
            if rng.random() < 0.3:  # 随机小角度旋转（镜像填充避免黑边伪影）
                h, w = img.shape[:2]
                angle = rng.uniform(-15.0, 15.0)
                mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                img = cv2.warpAffine(
                    np.ascontiguousarray(img),
                    mat,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
        return np.ascontiguousarray(img)

    def __getitem__(self, i: int) -> tuple[Any, int, str]:
        if torch is None:
            raise ImportError("读取图像张量需要安装 PyTorch（pip install torch）")

        img = load_bgr(self.files[i])[:, :, ::-1]  # BGR -> RGB
        img = center_crop_resize(img, size=self.size)
        label = self.labels[i]
        if self.augment != "none":
            img = self._aug(img, label)
        tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0
        mean_t = torch.tensor(MEAN, dtype=tensor.dtype)[:, None, None]
        std_t = torch.tensor(STD, dtype=tensor.dtype)[:, None, None]
        tensor = (tensor - mean_t) / std_t
        return tensor, label, self.keys[i]
