"""HAM10000 训练脚本。

基线：ResNet18 随机初始化 + 均匀采样 + 基础增强
改进：ResNet18 ImageNet 预训练 + 按类频率倒数加权采样 + 强增强

用法：
    python src/train.py --model resnet18_scratch --epochs 30
    python src/train.py --model resnet18_pretrained --weighted --augment strong --epochs 30
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np

from dataset import CLASSES, HAMDataset, compute_class_sample_weights

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    WeightedRandomSampler = None  # type: ignore[assignment]


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 随机种子以保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_out_dir(model_name: str, weighted: bool, out_dir: str | None = None) -> Path:
    """根据模型名与是否加权推导默认输出目录。"""
    if out_dir:
        return Path(out_dir)
    suffix = model_name.split("_", 1)[1] if "_" in model_name else model_name
    return Path(f"outputs/{suffix}{'_weighted' if weighted else ''}")


def validate_train_args(args: argparse.Namespace) -> None:
    """校验训练命令行超参数合法性。"""
    if args.epochs <= 0:
        raise ValueError(f"epochs 必须为正整数，当前为: {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，当前为: {args.batch_size}")
    if args.lr <= 0:
        raise ValueError(f"lr 必须大于 0，当前为: {args.lr}")
    if args.size <= 0:
        raise ValueError(f"size 必须为正整数，当前为: {args.size}")
    if args.workers < 0:
        raise ValueError(f"workers 不能为负数，当前为: {args.workers}")


def evaluate(model: Any, loader: Any, device: Any) -> tuple[float, float]:
    """在验证集上评估平均交叉熵损失与总体准确率。

    Args:
        model: 待评估神经网络模型。
        loader: 验证集 ``DataLoader``。
        device: 推理设备。

    Returns:
        二元组 ``(avg_loss, accuracy)``。

    Raises:
        ValueError: 验证集为空（无任何有效样本）。
    """
    if torch is None or nn is None:
        raise ImportError("运行评估需要安装 PyTorch")

    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += crit(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    if total == 0:
        raise ValueError("验证集为空，无法计算验证损失与准确率")
    return loss_sum / total, correct / total


def main() -> None:
    """CLI 入口：训练 HAM10000 皮肤病变分类模型并按验证准确率保存最优权重。"""
    if torch is None or nn is None or DataLoader is None or WeightedRandomSampler is None:
        raise ImportError("运行训练脚本需要安装 PyTorch（pip install torch torchvision）")

    from models import SUPPORTED_MODELS, build_model

    ap = argparse.ArgumentParser(description="HAM10000 ResNet18 训练脚本")
    ap.add_argument("--data-dir", default="data/ham10000")
    ap.add_argument("--model", choices=list(SUPPORTED_MODELS), default="resnet18_scratch")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weighted", action="store_true", help="训练时按类频率的倒数加权采样")
    ap.add_argument("--augment", choices=["none", "base", "strong"], default="base")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None, help="默认 outputs/<model 后缀>")
    args = ap.parse_args()
    validate_train_args(args)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve_out_dir(args.model, args.weighted, args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"设备: {device} | 模型: {args.model} | 加权采样: {args.weighted} | 增强: {args.augment}")

    csv_path = Path(args.data_dir) / "splits.csv"
    img_dir = Path(args.data_dir) / "images"
    train_set = HAMDataset(img_dir, csv_path, "train", augment=args.augment, size=args.size)
    val_set = HAMDataset(img_dir, csv_path, "val", augment="none", size=args.size)

    sampler = None
    if args.weighted:
        counts, class_w, sample_w = compute_class_sample_weights(
            train_set.labels, num_classes=len(CLASSES)
        )
        sampler = WeightedRandomSampler(
            sample_w.tolist(), num_samples=len(sample_w), replacement=True
        )
        max_w = float(class_w.max())
        norm_w = np.round(class_w / max_w, 3).tolist() if max_w > 0 else class_w.tolist()
        print("类别张数:", dict(zip(CLASSES, counts.tolist(), strict=True)))
        print("采样权重:", norm_w)

    # 样本数少于 batch_size 时自动关闭 drop_last，避免小样本冒烟训练时出现 0 批次除零
    drop_last = len(train_set) >= args.batch_size
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=128,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    model = build_model(args.model, num_classes=len(CLASSES)).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_csv = out_dir / "train_log.csv"
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        if total == 0:
            raise RuntimeError("训练 DataLoader 未产出任何批次，请检查数据集大小与 batch_size")

        sched.step()
        train_loss = loss_sum / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f}"
            f" | val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
        if val_acc > best:
            best = val_acc
            torch.save({"model": args.model, "state": model.state_dict()}, ckpt_dir / "best.pth")
    print(f"完成，最佳 val acc {best:.4f}，权重在 {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
