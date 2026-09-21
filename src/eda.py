"""EDA：类别分布图 + 样例图网格（报告用）。"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import HAMDataset, CLASSES, load_bgr

FULL_NAMES = {
    "akiec": "actinic keratosis", "bcc": "basal cell carcinoma", "bkl": "benign keratosis",
    "df": "dermatofibroma", "mel": "melanoma", "nv": "melanocytic nevus", "vasc": "vascular lesion",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ham10000")
    ap.add_argument("--out-dir", default="outputs/figures")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(Path(args.data_dir) / "splits.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(r["dx"] for r in rows)

    plt.figure(figsize=(7, 4))
    xs = range(len(CLASSES))
    plt.bar(xs, [counts[c] for c in CLASSES], color="#4c72b0")
    for i, c in enumerate(CLASSES):
        plt.text(i, counts[c] + 60, f"{counts[c]}\n({counts[c]/len(rows)*100:.1f}%)", ha="center", fontsize=8)
    plt.xticks(list(xs), [f"{c}\n{FULL_NAMES[c]}" for c in CLASSES], fontsize=8)
    plt.ylabel("image count")
    plt.title("HAM10000 class distribution (10015 images, 7 classes)")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=150)
    print(f"saved {out / 'class_distribution.png'}")

    ds = HAMDataset(Path(args.data_dir) / "images", Path(args.data_dir) / "splits.csv", "train")
    seen, idxs = set(), []
    for i, l in enumerate(ds.labels):
        if l not in seen:
            seen.add(l)
            idxs.append(i)
        if len(seen) == len(CLASSES):
            break
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, idxs):
        img = load_bgr(ds.files[i])[:, :, ::-1]
        ax.imshow(img)
        l = ds.labels[i]
        ax.set_title(f"{CLASSES[l]}: {FULL_NAMES[CLASSES[l]]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("One example per class")
    fig.tight_layout()
    fig.savefig(out / "class_examples.png", dpi=150)
    print(f"saved {out / 'class_examples.png'}")


if __name__ == "__main__":
    main()
