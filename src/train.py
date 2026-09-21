"""HAM10000 训练脚本。

基线：ResNet18 随机初始化 + 普通交叉熵 + 基础增强
改进：ResNet18 ImageNet 预训练 + 类别加权交叉熵 + 强增强

用法：
    python src/train.py --model resnet18_scratch --epochs 30
    python src/train.py --model resnet18_pretrained --weighted --augment strong --epochs 30
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import HAMDataset, read_splits, CLASSES
from models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss()
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ham10000")
    ap.add_argument("--model", choices=["resnet18_scratch", "resnet18_pretrained"], default="resnet18_scratch")
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

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir or f"outputs/{args.model.split('_', 1)[1]}{'_weighted' if args.weighted else ''}")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"设备: {device} | 模型: {args.model} | 加权采样: {args.weighted} | 增强: {args.augment}")

    csv_path = Path(args.data_dir) / "splits.csv"
    img_dir = Path(args.data_dir) / "images"
    train_set = HAMDataset(img_dir, csv_path, "train", augment=args.augment, size=args.size)
    val_set = HAMDataset(img_dir, csv_path, "val", augment="none", size=args.size)

    sampler = None
    if args.weighted:
        counts = np.bincount(train_set.labels, minlength=len(CLASSES))
        w = (1.0 / counts).clip(min=1e-6)
        weights = [w[l] for l in train_set.labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print("类别张数:", dict(zip(CLASSES, counts.tolist())))
        print("采样权重:", np.round(w / w.max(), 3).tolist())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                              shuffle=sampler is None, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=128, num_workers=args.workers)

    model = build_model(args.model).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_csv = out_dir / "train_log.csv"
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        sched.step()
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {loss_sum/total:.4f} acc {correct/total:.4f}"
              f" | val loss {val_loss:.4f} acc {val_acc:.4f}")
        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{loss_sum/total:.4f},{correct/total:.4f},{val_loss:.4f},{val_acc:.4f}\n")
        if val_acc > best:
            best = val_acc
            torch.save({"model": args.model, "state": model.state_dict()}, ckpt_dir / "best.pth")
    print(f"完成，最佳 val acc {best:.4f}，权重在 {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
