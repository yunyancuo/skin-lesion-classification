"""HAM10000 测试集评估：总体/平衡准确率、macro F1、每类召回、macro AUC、混淆矩阵图。

类别不平衡（nv 占 67%，df 只有 ~3%），总体准确率会被 nv 主导，
所以平衡准确率与每类召回是更重要的指标。

用法：
    python src/evaluate.py --weights outputs/pretrained_weighted/checkpoints/best.pth
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cn_font  # noqa: F401  注册中文字体
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from torch.utils.data import DataLoader

from dataset import HAMDataset, CLASSES
from models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", default="data/ham10000")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    model = build_model(ckpt["model"]).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    ds = HAMDataset(Path(args.data_dir) / "images", Path(args.data_dir) / "splits.csv", args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    save_dir = Path(args.save_dir or Path(args.weights).parents[1] / "eval")
    save_dir.mkdir(parents=True, exist_ok=True)

    probs_all, ys, keys = [], [], []
    with torch.no_grad():
        for x, y, k in loader:
            p = torch.softmax(model(x.to(device)), 1).cpu().numpy()
            probs_all.append(p)
            ys.append(y.numpy())
            keys += list(k)
    probs = np.concatenate(probs_all)
    ys = np.concatenate(ys)
    preds = probs.argmax(1)

    acc = (preds == ys).mean()
    bacc = balanced_accuracy_score(ys, preds)
    macro_f1 = f1_score(ys, preds, average="macro")
    try:
        auc = roc_auc_score(ys, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    report = classification_report(ys, preds, target_names=CLASSES, digits=4, zero_division=0)

    print(f"=== {args.split} 集（n={len(ys)}）===")
    print(f"accuracy        : {acc:.4f}")
    print(f"balanced acc    : {bacc:.4f}")
    print(f"macro F1        : {macro_f1:.4f}")
    print(f"macro AUC (OvR) : {auc:.4f}")
    print("\n每类指标：")
    print(report)

    with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"accuracy {acc:.4f}\nbalanced_acc {bacc:.4f}\nmacro_f1 {macro_f1:.4f}\nmacro_auc {auc:.4f}\n\n")
        f.write(report)
    np.savez(save_dir / "probs.npz", probs=probs, ys=ys, preds=preds, keys=np.array(keys))

    cm = confusion_matrix(ys, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"混淆矩阵（行归一化），平衡准确率 {bacc:.3f}")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=150)
    print(f"\n结果保存到 {save_dir}")


if __name__ == "__main__":
    main()
