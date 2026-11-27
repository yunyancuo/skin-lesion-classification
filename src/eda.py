"""EDA：类别分布柱状图与每类样例图网格（报告用，中文标签）。"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cn_font  # noqa: F401  注册中文字体
from dataset import CLASSES, load_bgr, read_splits

CN_NAMES: dict[str, str] = {
    "akiec": "光化性角化病",
    "bcc": "基底细胞癌",
    "bkl": "良性角化病",
    "df": "皮肤纤维瘤",
    "mel": "黑色素瘤",
    "nv": "黑色素细胞痣",
    "vasc": "血管性病损",
}


def plot_class_distribution(
    rows: Sequence[Mapping[str, str]],
    out_path: str | Path,
) -> Path:
    """绘制 7 类病变样本分布柱状图并释放 Matplotlib 内存。

    Args:
        rows: 包含 ``dx`` 字段的数据记录列表。
        out_path: 图片输出路径。

    Returns:
        保存后的图片路径。

    Raises:
        ValueError: ``rows`` 为空或全部类别计数为 0。
    """
    if not rows:
        raise ValueError("splits.csv 为空，无法绘制类别分布图")

    counts = Counter(row["dx"] for row in rows if row.get("dx") in CN_NAMES)
    max_count = max((counts[c] for c in CLASSES), default=0)
    if max_count == 0:
        raise ValueError("未统计到任何合法的 HAM10000 类别样本")

    total = len(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    try:
        xs = list(range(len(CLASSES)))
        ax.bar(xs, [counts[c] for c in CLASSES], color="#4c72b0")
        for idx, cls in enumerate(CLASSES):
            # 单行标注，柱顶预留足够高度避免相邻文字重叠
            ax.text(
                idx,
                counts[cls] + max_count * 0.03,
                f"{counts[cls]}（{counts[cls] / total * 100:.1f}%）",
                ha="center",
                fontsize=8.5,
            )
        ax.set_xticks(xs, [f"{c}\n{CN_NAMES[c]}" for c in CLASSES], fontsize=9)
        ax.set_ylim(0, max_count * 1.22)
        ax.set_ylabel("图像数（张）")
        ax.set_title(f"HAM10000 类别分布（共 {total} 张，7 类）")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
    finally:
        plt.close(fig)
    return out


def plot_class_examples(
    rows: Sequence[Mapping[str, str]],
    img_dir: str | Path,
    out_path: str | Path,
    split: str = "train",
) -> Path:
    """为每个类别抽取一张样例图绘制 2×4 网格并释放 Matplotlib 内存。"""
    split_rows = [r for r in rows if r.get("split") == split]
    if not split_rows:
        raise ValueError(f"在 splits.csv 中未找到 split={split!r} 的样本")

    seen: set[int] = set()
    picked: list[tuple[Path, int]] = []
    base = Path(img_dir)
    for row in split_rows:
        label = int(row["label"])
        if label not in seen:
            seen.add(label)
            picked.append((base / f"{row['image_id']}.jpg", label))
        if len(seen) == len(CLASSES):
            break

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    try:
        for ax in axes.flat:
            ax.axis("off")
        for ax, (img_path, label) in zip(axes.flat, picked, strict=False):
            img = load_bgr(img_path)[:, :, ::-1]
            ax.imshow(img)
            cls = CLASSES[label]
            ax.set_title(f"{cls}：{CN_NAMES[cls]}", fontsize=9)
            ax.axis("off")
        fig.suptitle("每类一张样例")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
    finally:
        plt.close(fig)
    return out


def main() -> None:
    """CLI 入口：生成 HAM10000 类别分布图与每类样例网格图。"""
    ap = argparse.ArgumentParser(description="HAM10000 探索性数据分析出图")
    ap.add_argument("--data-dir", default="data/ham10000")
    ap.add_argument("--out-dir", default="outputs/figures")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_splits(data_dir / "splits.csv")
    dist_path = plot_class_distribution(rows, out / "class_distribution.png")
    print(f"saved {dist_path}")

    examples_path = plot_class_examples(rows, data_dir / "images", out / "class_examples.png")
    print(f"saved {examples_path}")


if __name__ == "__main__":
    main()
