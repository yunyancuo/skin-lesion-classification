"""生成测试结果附件：逐图预测明细 CSV + 预测示例图（对/错各若干张）。

用法：
    python src/make_test_results.py --weights outputs/pretrained_weighted/checkpoints/best.pth \
        --out-dir results
输出：
    results/test_predictions.csv      # image_id, lesion_id, 真实类, 预测类, 预测置信度, 是否正确
    results/prediction_examples.png   # 2×4 示例：上排判对，下排判错
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cn_font  # noqa: F401  注册中文字体
from dataset import CLASSES, MEAN, STD, HAMDataset
from evaluate import load_checkpoint_safe

CLINICAL_HIGH_RISK_CLASSES: tuple[str, ...] = ("mel", "bcc", "akiec")


def denormalize_image(img_hwc: np.ndarray) -> np.ndarray:
    """将按 ImageNet 标准化后的 ``(H, W, 3)`` 图像反归一化到 ``[0, 1]`` 区间。"""
    mean = np.asarray(MEAN, dtype=np.float64)[None, None, :]
    std = np.asarray(STD, dtype=np.float64)[None, None, :]
    return np.clip(img_hwc * std + mean, 0.0, 1.0)


def write_predictions_csv(
    out_csv: str | Path,
    keys: Sequence[str],
    lesion_of: Mapping[str, str],
    ys: np.ndarray,
    preds: np.ndarray,
    conf: np.ndarray,
    classes: Sequence[str] = CLASSES,
) -> float:
    """写出逐图预测明细 CSV（UTF-8-SIG 编码，Excel 友好）并返回总体准确率。"""
    n = len(keys)
    if n == 0:
        raise ValueError("测试预测结果为空，无法写入 CSV")
    if not (len(ys) == len(preds) == len(conf) == n):
        raise ValueError("keys、ys、preds、conf 长度必须完全一致")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "lesion_id", "真实类", "预测类", "预测置信度", "是否正确"])
        for idx in range(n):
            ok = "正确" if int(preds[idx]) == int(ys[idx]) else "错误"
            writer.writerow(
                [
                    keys[idx],
                    lesion_of.get(keys[idx], ""),
                    classes[int(ys[idx])],
                    classes[int(preds[idx])],
                    f"{float(conf[idx]):.3f}",
                    ok,
                ]
            )
    return float(np.mean(np.asarray(preds) == np.asarray(ys)))


def select_example_indices(
    ys: np.ndarray,
    preds: np.ndarray,
    classes: Sequence[str] = CLASSES,
    n_right: int = 4,
    n_wrong: int = 4,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """安全抽样判对与判错示例索引（优先涵盖临床高危恶性/癌前错例，防护小样本不足）。

    Args:
        ys: 真实类别索引数组。
        preds: 预测类别索引数组。
        classes: 类别名称列表。
        n_right: 期望抽取的判对样例上限。
        n_wrong: 期望抽取的判错样例上限。
        seed: 抽样随机种子。

    Returns:
        二元组 ``(right_pick, wrong_pick)``。
    """
    y_arr = np.asarray(ys, dtype=np.int64)
    p_arr = np.asarray(preds, dtype=np.int64)
    rng = np.random.default_rng(seed)

    right = np.where(p_arr == y_arr)[0]
    wrong = np.where(p_arr != y_arr)[0]

    k_right = min(max(n_right, 0), len(right))
    right_pick = (
        [int(i) for i in rng.choice(right, size=k_right, replace=False)] if k_right > 0 else []
    )

    k_wrong = min(max(n_wrong, 0), len(wrong))
    if k_wrong == 0:
        return right_pick, []

    clin = [int(i) for i in wrong if classes[int(y_arr[i])] in CLINICAL_HIGH_RISK_CLASSES]
    other_wrong = [int(i) for i in wrong if int(i) not in set(clin)]

    # 优先挑选最多 min(2, k_wrong) 个临床高危错例，其余从非高危错例补齐
    n_clin = min(2, k_wrong, len(clin))
    wrong_pick = list(clin[:n_clin])
    remain = k_wrong - len(wrong_pick)

    if remain > 0 and other_wrong:
        take_other = min(remain, len(other_wrong))
        wrong_pick.extend(int(i) for i in rng.choice(other_wrong, size=take_other, replace=False))
        remain = k_wrong - len(wrong_pick)

    if remain > 0 and len(clin) > n_clin:
        extra_clin = clin[n_clin:]
        wrong_pick.extend(extra_clin[:remain])

    return right_pick, wrong_pick


def plot_prediction_examples(
    image_loader: Callable[[int], np.ndarray],
    ys: np.ndarray,
    preds: np.ndarray,
    conf: np.ndarray,
    right_pick: Sequence[int],
    wrong_pick: Sequence[int],
    out_path: str | Path,
    classes: Sequence[str] = CLASSES,
) -> Path:
    """绘制 2×4 测试集预测对/错示例图并确保释放 Matplotlib 内存。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.6))
    try:
        for ax in axes.flat:
            ax.axis("off")
        for ax, idx in zip(axes[0], right_pick, strict=False):
            ax.imshow(image_loader(idx))
            true_cls = classes[int(ys[idx])]
            pred_cls = classes[int(preds[idx])]
            ax.set_title(
                f"真实 {true_cls} / 预测 {pred_cls} [正确]",
                fontsize=10,
                color="#1a7a3a",
            )
        for ax, idx in zip(axes[1], wrong_pick, strict=False):
            ax.imshow(image_loader(idx))
            true_cls = classes[int(ys[idx])]
            pred_cls = classes[int(preds[idx])]
            ax.set_title(
                f"真实 {true_cls} / 预测 {pred_cls} [错误] (p={conf[idx]:.2f})",
                fontsize=10,
                color="#b03030",
            )
        fig.suptitle("测试集预测示例：上排判对，下排判错（含恶性类典型错误）", fontsize=12)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
    finally:
        plt.close(fig)
    return out


def main() -> None:
    """CLI 入口：导出测试集逐图预测 CSV 与预测对/错示例图。"""
    import torch
    from torch.utils.data import DataLoader

    from models import build_model

    ap = argparse.ArgumentParser(description="生成测试集逐图预测明细 CSV 与预测示例图")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", default="data/ham10000")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint_safe(args.weights, device)
    model = build_model(ckpt["model"], num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    splits_csv = Path(args.data_dir) / "splits.csv"
    ds = HAMDataset(Path(args.data_dir) / "images", splits_csv, "test")
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    probs_list: list[np.ndarray] = []
    ys_list: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, _ in loader:
            probs_list.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
            ys_list.append(y.numpy())

    if not probs_list:
        raise ValueError("测试集为空，无法生成预测结果")

    probs = np.concatenate(probs_list)
    ys = np.concatenate(ys_list)
    preds = probs.argmax(1)
    conf = probs.max(1)

    lesion_of: dict[str, str] = {}
    with open(splits_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lesion_of[row["image_id"]] = row["lesion_id"]

    acc = write_predictions_csv(
        out / "test_predictions.csv",
        keys=ds.keys,
        lesion_of=lesion_of,
        ys=ys,
        preds=preds,
        conf=conf,
        classes=CLASSES,
    )
    print(f"CSV: {len(ds)} 张，总体准确率 {acc:.4f}")

    right_pick, wrong_pick = select_example_indices(ys, preds, classes=CLASSES, seed=42)
    example_fig = plot_prediction_examples(
        image_loader=lambda idx: denormalize_image(ds[idx][0].numpy().transpose(1, 2, 0)),
        ys=ys,
        preds=preds,
        conf=conf,
        right_pick=right_pick,
        wrong_pick=wrong_pick,
        out_path=out / "prediction_examples.png",
        classes=CLASSES,
    )
    print(f"示例图: {example_fig}")


if __name__ == "__main__":
    main()
