"""HAM10000 测试集评估：总体/平衡准确率、macro F1、每类召回、macro AUC、混淆矩阵图。

类别不平衡（nv 占 67%，df 仅约 1.1%），总体准确率会被 nv 主导，
所以平衡准确率、Macro F1 与每类召回是更关键的临床评价指标。

用法：
    python src/evaluate.py --weights outputs/pretrained_weighted/checkpoints/best.pth
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cn_font  # noqa: F401  注册中文字体
from dataset import CLASSES, HAMDataset

try:
    from sklearn.metrics import (
        balanced_accuracy_score as _sk_balanced_accuracy_score,
    )
    from sklearn.metrics import (
        classification_report as _sk_classification_report,
    )
    from sklearn.metrics import (
        confusion_matrix as _sk_confusion_matrix,
    )
    from sklearn.metrics import (
        f1_score as _sk_f1_score,
    )
    from sklearn.metrics import (
        roc_auc_score as _sk_roc_auc_score,
    )

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """计算一维数组的 1-based 平均秩（与 Mann-Whitney U / ROC-AUC 并列处理一致）。"""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    n = len(sorted_vals)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _binary_roc_auc(y_true_bin: np.ndarray, scores: np.ndarray) -> float:
    """使用平均秩公式精确计算二分类 ROC-AUC。"""
    n_pos = int(np.sum(y_true_bin == 1))
    n_neg = int(len(y_true_bin) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    pos_rank_sum = float(np.sum(ranks[y_true_bin == 1]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_confusion_matrix(
    ys: np.ndarray,
    preds: np.ndarray,
    num_classes: int = len(CLASSES),
    normalize: str | None = None,
) -> np.ndarray:
    """计算固定维度 ``(num_classes, num_classes)`` 的混淆矩阵（防护缺失类别越界）。

    Args:
        ys: 真实整型标签数组。
        preds: 预测整型标签数组。
        num_classes: 类别总数。
        normalize: 可选归一化模式，支持 ``None`` 与 ``"true"``（行归一化）。

    Returns:
        形状恒为 ``(num_classes, num_classes)`` 的混淆矩阵。
    """
    if normalize not in (None, "true"):
        raise ValueError(f"不支持的 normalize 参数: {normalize!r}，仅支持 None 或 'true'")

    y_arr = np.asarray(ys, dtype=np.int64)
    p_arr = np.asarray(preds, dtype=np.int64)
    if y_arr.size == 0 or p_arr.size == 0 or y_arr.shape != p_arr.shape:
        raise ValueError("ys 与 preds 必须为非空且形状相同的一维数组")
    if (
        np.any(y_arr < 0)
        or np.any(y_arr >= num_classes)
        or np.any(p_arr < 0)
        or np.any(p_arr >= num_classes)
    ):
        raise ValueError(f"标签越界，合法类别范围为 0..{num_classes - 1}")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_arr, p_arr), 1)
    if normalize is None:
        return cm

    cm_float = cm.astype(np.float64)
    row_sums = cm_float.sum(axis=1, keepdims=True)
    # 缺失类别（行和为 0）安全置 0.0，避免 NaN 污染混淆矩阵可视化
    return np.divide(cm_float, row_sums, out=np.zeros_like(cm_float), where=row_sums > 0)


def format_classification_report(
    cm_raw: np.ndarray,
    classes: Sequence[str] = CLASSES,
    digits: int = 4,
) -> str:
    """基于原始混淆矩阵格式化输出与 scikit-learn 风格一致的分类报告。"""
    num_classes = len(classes)
    precisions = np.zeros(num_classes, dtype=np.float64)
    recalls = np.zeros(num_classes, dtype=np.float64)
    f1s = np.zeros(num_classes, dtype=np.float64)
    supports = cm_raw.sum(axis=1)
    pred_totals = cm_raw.sum(axis=0)

    for idx in range(num_classes):
        tp = float(cm_raw[idx, idx])
        prec = tp / pred_totals[idx] if pred_totals[idx] > 0 else 0.0
        rec = tp / supports[idx] if supports[idx] > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions[idx] = prec
        recalls[idx] = rec
        f1s[idx] = f1

    total = int(supports.sum())
    acc = float(np.trace(cm_raw) / total) if total > 0 else 0.0
    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    if total > 0:
        weights = supports / total
        weighted_p = float(np.sum(precisions * weights))
        weighted_r = float(np.sum(recalls * weights))
        weighted_f1 = float(np.sum(f1s * weights))
    else:
        weighted_p = weighted_r = weighted_f1 = 0.0

    name_width = max(max(len(c) for c in classes), len("weighted avg"))
    header = f"{'':>{name_width}}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>9}"
    lines = [header, ""]
    for idx, cls in enumerate(classes):
        lines.append(
            f"{cls:>{name_width}}  {precisions[idx]:9.{digits}f}  "
            f"{recalls[idx]:9.{digits}f}  {f1s[idx]:9.{digits}f}  {int(supports[idx]):9d}"
        )
    lines.append("")
    lines.append(
        f"{'accuracy':>{name_width}}  {'':>9}  {'':>9}  {acc:9.{digits}f}  {total:9d}"
    )
    lines.append(
        f"{'macro avg':>{name_width}}  {macro_p:9.{digits}f}  "
        f"{macro_r:9.{digits}f}  {macro_f1:9.{digits}f}  {total:9d}"
    )
    lines.append(
        f"{'weighted avg':>{name_width}}  {weighted_p:9.{digits}f}  "
        f"{weighted_r:9.{digits}f}  {weighted_f1:9.{digits}f}  {total:9d}"
    )
    return "\n".join(lines)


def _compute_metrics_numpy(
    ys: np.ndarray,
    probs: np.ndarray,
    classes: Sequence[str] = CLASSES,
) -> dict[str, Any]:
    """纯 NumPy 精确评估引擎（与 scikit-learn 数学等价，支持离线无 sklearn 环境）。"""
    num_classes = len(classes)
    preds = probs.argmax(axis=1)
    cm_raw = compute_confusion_matrix(ys, preds, num_classes=num_classes, normalize=None)
    cm_norm = compute_confusion_matrix(ys, preds, num_classes=num_classes, normalize="true")

    acc = float(np.mean(preds == ys))
    supports = cm_raw.sum(axis=1)
    pred_totals = cm_raw.sum(axis=0)

    present_true = supports > 0
    recalls_true = np.divide(
        np.diag(cm_raw).astype(np.float64),
        supports.astype(np.float64),
        out=np.zeros(num_classes, dtype=np.float64),
        where=present_true,
    )
    bacc = float(np.mean(recalls_true[present_true])) if np.any(present_true) else float("nan")

    # Macro F1：与 sklearn 默认行为一致，在真实或预测出现过的类别上取平均
    present_union = (supports > 0) | (pred_totals > 0)
    f1s = np.zeros(num_classes, dtype=np.float64)
    for idx in range(num_classes):
        tp = float(cm_raw[idx, idx])
        prec = tp / pred_totals[idx] if pred_totals[idx] > 0 else 0.0
        rec = tp / supports[idx] if supports[idx] > 0 else 0.0
        f1s[idx] = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    macro_f1 = float(np.mean(f1s[present_union])) if np.any(present_union) else 0.0

    # Macro OvR AUC：仅当全部类别均有正样本与负样本且概率全为有限值时计算
    if (
        np.all(np.isfinite(probs))
        and probs.shape[1] == num_classes
        and int(np.sum(present_true)) == num_classes
    ):
        aucs = [
            _binary_roc_auc((ys == idx).astype(np.int64), probs[:, idx])
            for idx in range(num_classes)
        ]
        macro_auc = float(np.mean(aucs)) if all(np.isfinite(aucs)) else float("nan")
    else:
        macro_auc = float("nan")

    report = format_classification_report(cm_raw, classes=classes, digits=4)
    return {
        "accuracy": acc,
        "balanced_acc": bacc,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "preds": preds,
        "cm_raw": cm_raw,
        "cm_norm": cm_norm,
        "report": report,
    }


def compute_classification_metrics(
    ys: np.ndarray,
    probs: np.ndarray,
    classes: Sequence[str] = CLASSES,
    use_sklearn: bool = True,
) -> dict[str, Any]:
    """计算多分类完整指标：Accuracy、Balanced Accuracy、Macro F1、Macro AUC 与混淆矩阵。

    对缺失类别、空样本与不可计算 AUC 等边界场景做完整防护。

    Args:
        ys: 真实整型标签，形状 ``(N,)``。
        probs: 类别预测概率矩阵，形状 ``(N, C)``。
        classes: 类别名称列表，长度为 ``C``。
        use_sklearn: 在 scikit-learn 可用时是否优先调用其官方实现。

    Returns:
        包含 ``accuracy``、``balanced_acc``、``macro_f1``、``macro_auc``、``preds``、
        ``cm_raw``、``cm_norm`` 与 ``report`` 的字典。

    Raises:
        ValueError: 输入为空、维度不匹配或标签越界。
    """
    y_arr = np.asarray(ys, dtype=np.int64)
    p_arr = np.asarray(probs, dtype=np.float64)
    num_classes = len(classes)
    if num_classes == 0:
        raise ValueError("classes 不能为空")
    if y_arr.ndim != 1 or y_arr.size == 0:
        raise ValueError("ys 必须为非空一维数组")
    if p_arr.ndim != 2 or p_arr.shape != (y_arr.size, num_classes):
        raise ValueError(
            f"probs 形状应为 ({y_arr.size}, {num_classes})，当前为 {p_arr.shape}"
        )
    if np.any(y_arr < 0) or np.any(y_arr >= num_classes):
        raise ValueError(f"ys 包含越界标签，合法范围为 0..{num_classes - 1}")

    if not (use_sklearn and _HAS_SKLEARN):
        return _compute_metrics_numpy(y_arr, p_arr, classes=classes)

    preds = p_arr.argmax(axis=1)
    all_labels = list(range(num_classes))
    acc = float(np.mean(preds == y_arr))
    bacc = float(_sk_balanced_accuracy_score(y_arr, preds))
    macro_f1 = float(_sk_f1_score(y_arr, preds, average="macro", zero_division=0))
    try:
        macro_auc = float(_sk_roc_auc_score(y_arr, p_arr, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")

    report = _sk_classification_report(
        y_arr,
        preds,
        labels=all_labels,
        target_names=list(classes),
        digits=4,
        zero_division=0,
    )
    cm_raw = _sk_confusion_matrix(y_arr, preds, labels=all_labels)
    cm_norm = compute_confusion_matrix(y_arr, preds, num_classes=num_classes, normalize="true")
    return {
        "accuracy": acc,
        "balanced_acc": bacc,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "preds": preds,
        "cm_raw": cm_raw,
        "cm_norm": cm_norm,
        "report": report,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Sequence[str],
    bacc: float,
    out_path: str | Path,
) -> Path:
    """绘制行归一化混淆矩阵热力图并确保释放 Matplotlib 内存。"""
    num_classes = len(classes)
    if cm.shape != (num_classes, num_classes):
        raise ValueError(f"混淆矩阵形状 {cm.shape} 与类别数 {num_classes} 不匹配")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    try:
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes), list(classes), rotation=45)
        ax.set_yticks(range(num_classes), list(classes))
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > 0.5 else "black",
                    fontsize=8,
                )
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(f"混淆矩阵（行归一化），平衡准确率 {bacc:.3f}")
        fig.colorbar(im, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
    finally:
        plt.close(fig)
    return out


def resolve_eval_save_dir(weights_path: str | Path, save_dir: str | Path | None = None) -> Path:
    """推导评估结果保存目录（兼容顶层相对路径，避免 ``parents[1]`` 越界）。"""
    if save_dir is not None:
        return Path(save_dir)
    resolved = Path(weights_path).resolve()
    parent = resolved.parents[1] if len(resolved.parents) > 1 else resolved.parent
    return parent / "eval"


def load_checkpoint_safe(weights_path: str | Path, device: Any) -> dict[str, Any]:
    """安全加载 PyTorch checkpoint，优先使用 ``weights_only=True``。"""
    import torch

    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(weights_path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "state" not in ckpt:
        raise ValueError(f"checkpoint 格式不合法（缺少 model 或 state 字段）: {weights_path}")
    return ckpt


def main() -> None:
    """CLI 入口：加载训练权重并在指定 split 上执行全量指标评估与混淆矩阵绘图。"""
    import torch
    from torch.utils.data import DataLoader

    from models import build_model

    ap = argparse.ArgumentParser(description="HAM10000 皮肤病变分类测试集评估")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", default="data/ham10000")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint_safe(args.weights, device)
    model = build_model(ckpt["model"], num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    ds = HAMDataset(Path(args.data_dir) / "images", Path(args.data_dir) / "splits.csv", args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    save_dir = resolve_eval_save_dir(args.weights, args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    probs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    keys: list[str] = []
    with torch.no_grad():
        for x, y, k in loader:
            p = torch.softmax(model(x.to(device)), 1).cpu().numpy()
            probs_all.append(p)
            ys_all.append(y.numpy())
            keys.extend(k)

    if not probs_all:
        raise ValueError(f"split={args.split!r} 中没有任何可评估样本")

    probs = np.concatenate(probs_all)
    ys = np.concatenate(ys_all)
    metrics = compute_classification_metrics(ys, probs, classes=CLASSES)

    acc = metrics["accuracy"]
    bacc = metrics["balanced_acc"]
    macro_f1 = metrics["macro_f1"]
    auc = metrics["macro_auc"]
    preds = metrics["preds"]
    report = metrics["report"]
    cm = metrics["cm_norm"]

    print(f"=== {args.split} 集（n={len(ys)}）===")
    print(f"accuracy        : {acc:.4f}")
    print(f"balanced acc    : {bacc:.4f}")
    print(f"macro F1        : {macro_f1:.4f}")
    print(f"macro AUC (OvR) : {auc:.4f}")
    print("\n每类指标：")
    print(report)

    with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(
            f"accuracy {acc:.4f}\nbalanced_acc {bacc:.4f}\n"
            f"macro_f1 {macro_f1:.4f}\nmacro_auc {auc:.4f}\n\n"
        )
        f.write(report)
    np.savez(save_dir / "probs.npz", probs=probs, ys=ys, preds=preds, keys=np.array(keys))
    plot_confusion_matrix(cm, CLASSES, bacc, save_dir / "confusion_matrix.png")
    print(f"\n结果保存到 {save_dir}")


if __name__ == "__main__":
    main()
