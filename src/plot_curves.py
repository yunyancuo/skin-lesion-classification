"""画 HAM10000 训练曲线：loss 与 acc（train/val）。

用法：
    python src/plot_curves.py --logs outputs/scratch/train_log.csv outputs/pretrained_weighted/train_log.csv \
        --names "ResNet18 (scratch)" "ResNet18 (pretrained+weighted)" --out outputs/figures/train_curves.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_log(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d["epoch"], d["train_loss"], d["train_acc"], d["val_loss"], d["val_acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/figures/train_curves.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for log, name in zip(args.logs, args.names):
        ep, tl, ta, vl, va = read_log(log)
        axes[0].plot(ep, tl, label=name)
        axes[0].plot(ep, vl, label=name + " (val)", linestyle="--", alpha=0.6)
        axes[1].plot(ep, ta, label=name)
        axes[1].plot(ep, va, label=name + " (val)", linestyle="--", alpha=0.6)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy (overall)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
