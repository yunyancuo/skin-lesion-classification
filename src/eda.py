"""EDA：类别分布图 + 样例图网格（报告用，中文标签）。"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cn_font  # noqa: F401  注册中文字体
import matplotlib.pyplot as plt

from dataset import HAMDataset, CLASSES, load_bgr

CN_NAMES = {
    "akiec": "光化性角化病", "bcc": "基底细胞癌", "bkl": "良性角化病",
    "df": "皮肤纤维瘤", "mel": "黑色素瘤", "nv": "黑色素细胞痣", "vasc": "血管性病损",
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

    plt.figure(figsize=(7.5, 4.5))
    xs = range(len(CLASSES))
    bars = plt.bar(xs, [counts[c] for c in CLASSES], color="#4c72b0")
    for i, c in enumerate(CLASSES):
        # 单行标注，柱顶预留 25% 高度避免与相邻标注相碰
        plt.text(i, counts[c] + counts["nv"] * 0.03,
                 f"{counts[c]}（{counts[c]/len(rows)*100:.1f}%）", ha="center", fontsize=8.5)
    plt.xticks(list(xs), [f"{c}\n{CN_NAMES[c]}" for c in CLASSES], fontsize=9)
    plt.ylim(0, counts["nv"] * 1.22)
    plt.ylabel("图像数（张）")
    plt.title(f"HAM10000 类别分布（共 {len(rows)} 张，7 类）")
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
        ax.set_title(f"{CLASSES[l]}：{CN_NAMES[CLASSES[l]]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("每类一张样例")
    fig.tight_layout()
    fig.savefig(out / "class_examples.png", dpi=150)
    print(f"saved {out / 'class_examples.png'}")


if __name__ == "__main__":
    main()
