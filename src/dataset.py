"""HAM10000 数据集：读取 prepare.py 生成的 splits.csv 与图像。

基线与改进共用同一数据接口，差别只在增强强度（improved 用更强的颜色扰动）。
"""
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet 统计


def load_bgr(path):
    m = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if m is None:
        raise IOError(f"图像读取失败: {path}")
    return m


def read_splits(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class HAMDataset(Dataset):
    """img_dir 为图像目录，split 为 train/val/test。"""

    def __init__(self, img_dir, splits_csv, split, augment="none", size=224):
        self.dir = Path(img_dir)
        rows = [r for r in read_splits(splits_csv) if r["split"] == split]
        self.files = [self.dir / f"{r['image_id']}.jpg" for r in rows]
        self.labels = [int(r["label"]) for r in rows]
        self.keys = [r["image_id"] for r in rows]
        self.augment = augment
        self.size = size

    def __len__(self):
        return len(self.files)

    def _aug(self, img, label):
        rng = random
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.5:
            img = img[::-1]
        k = rng.randint(0, 3)
        img = np.rot90(img, k)
        if self.augment == "strong":
            if rng.random() < 0.8:  # 皮肤镜图像颜色差异大，做颜色扰动
                img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-20, 20))
            if rng.random() < 0.5:
                img = cv2.GaussianBlur(img, (3, 3), 0)
            if rng.random() < 0.3:  # 随机小角度旋转 + 裁剪
                img = np.ascontiguousarray(np.rot90(img, rng.randint(0, 3)))
        return np.ascontiguousarray(img)

    def __getitem__(self, i):
        img = load_bgr(self.files[i])[:, :, ::-1]  # BGR -> RGB
        h, w = img.shape[:2]
        s = min(h, w)
        img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]  # 中心方裁
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        label = self.labels[i]
        if self.augment != "none":
            img = self._aug(img, label)
        t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        t = (t - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
        return t, label, self.keys[i]
