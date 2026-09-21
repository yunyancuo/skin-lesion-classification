"""生成测试结果附件：逐图预测明细 CSV + 预测示例图（对/错各若干张）。

用法：
    python src/make_test_results.py --weights outputs/pretrained_weighted/checkpoints/best.pth \
        --out-dir results
输出：
    results/test_predictions.csv      # image_id, lesion_id, 真实类, 预测类, 置信度, 是否正确
    results/prediction_examples.png   # 2×4 示例：上排判对，下排判错
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HAMDataset, CLASSES
from models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", default="data/ham10000")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    model = build_model(ckpt["model"]).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    ds = HAMDataset(Path(args.data_dir) / "images", Path(args.data_dir) / "splits.csv", "test")
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=8)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    probs, ys = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
            ys.append(y.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    preds = probs.argmax(1)
    conf = probs.max(1)

    # 1) 逐图明细 CSV（lesion_id 从 splits.csv 查表）
    lesion_of = {}
    with open(Path(args.data_dir) / "splits.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lesion_of[r["image_id"]] = r["lesion_id"]
    with open(out / "test_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "lesion_id", "真实类", "预测类", "预测置信度", "是否正确"])
        for i in range(len(ds)):
            ok = "正确" if preds[i] == ys[i] else "错误"
            w.writerow([ds.keys[i], lesion_of.get(ds.keys[i], ""), CLASSES[ys[i]], CLASSES[preds[i]],
                        f"{conf[i]:.3f}", ok])
    acc = (preds == ys).mean()
    print(f"CSV: {len(ds)} 张，总体准确率 {acc:.4f}")

    # 2) 示例图：判对挑置信度适中的（更能代表真实表现），判错挑临床重要类（mel/bcc/akiec）
    rng = np.random.default_rng(42)
    right = np.where(preds == ys)[0]
    wrong = np.where(preds != ys)[0]
    clin = [i for i in wrong if CLASSES[ys[i]] in ("mel", "bcc", "akiec")]
    right_pick = list(rng.choice(right, size=4, replace=False))
    wrong_pick = clin[:2] + list(rng.choice([i for i in wrong if i not in clin], size=2, replace=False))

    def denorm(t):
        mean = np.array([0.485, 0.456, 0.406])[None, None, :]
        std = np.array([0.229, 0.224, 0.225])[None, None, :]
        return np.clip(t * std + mean, 0, 1)

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes[0], right_pick):
        img = denorm(ds[i][0].numpy().transpose(1, 2, 0))
        ax.imshow(img)
        ax.set_title(f"真实 {CLASSES[ys[i]]} / 预测 {CLASSES[preds[i]]} ✓", fontsize=10, color="#1a7a3a")
    for ax, i in zip(axes[1], wrong_pick):
        img = denorm(ds[i][0].numpy().transpose(1, 2, 0))
        ax.imshow(img)
        ax.set_title(f"真实 {CLASSES[ys[i]]} / 预测 {CLASSES[preds[i]]} ✗ (p={conf[i]:.2f})", fontsize=10, color="#b03030")
    fig.suptitle("测试集预测示例：上排判对，下排判错（含恶性类典型错误）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "prediction_examples.png", dpi=150)
    print(f"示例图: {out / 'prediction_examples.png'}")


if __name__ == "__main__":
    main()
