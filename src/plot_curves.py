"""绘制 HAM10000 训练曲线：训练/验证损失与准确率对比图。

用法：
    python src/plot_curves.py \\
        --logs outputs/scratch/train_log.csv outputs/pretrained_weighted/train_log.csv \\
        --names "ResNet18 (scratch)" "ResNet18 (pretrained+weighted)" \\
        --out outputs/figures/train_curves.png
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cn_font  # noqa: F401  注册中文字体

REQUIRED_COLUMNS: tuple[str, ...] = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc")


def read_log(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """安全解析训练日志 CSV（兼容单行 1-epoch 日志，并校验必需字段与空数据）。

    Args:
        path: ``train_log.csv`` 路径。

    Returns:
        五元组 ``(epoch, train_loss, train_acc, val_loss, val_acc)``，均为一维 float64 数组。

    Raises:
        ValueError: CSV 缺少必需列或不包含任何数据行。
    """
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(f"训练日志 {path} 缺少必需列: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"训练日志 {path} 为空，无法绘制训练曲线")

    return tuple(
        np.array([float(r[col]) for r in rows], dtype=np.float64) for col in REQUIRED_COLUMNS
    )


def plot_training_curves(
    logs: Sequence[str | Path],
    names: Sequence[str],
    out_path: str | Path,
) -> Path:
    """绘制多组实验的损失与准确率曲线并释放 Matplotlib 内存。

    Args:
        logs: 训练日志 CSV 路径列表。
        names: 对应的图例名称列表，长度必须与 ``logs`` 相同。
        out_path: 输出图片路径。

    Returns:
        保存后的图片路径。

    Raises:
        ValueError: ``logs`` 为空或与 ``names`` 数量不匹配。
    """
    if not logs:
        raise ValueError("logs 不能为空")
    if len(logs) != len(names):
        raise ValueError(
            f"--logs 与 --names 数量不一致（收到 {len(logs)} 个日志与 {len(names)} 个名称）"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    try:
        for log, name in zip(logs, names, strict=True):
            ep, tl, ta, vl, va = read_log(log)
            axes[0].plot(ep, tl, label=name)
            axes[0].plot(ep, vl, label=f"{name}（验证）", linestyle="--", alpha=0.6)
            axes[1].plot(ep, ta, label=name)
            axes[1].plot(ep, va, label=f"{name}（验证）", linestyle="--", alpha=0.6)

        axes[0].set_xlabel("训练轮次")
        axes[0].set_ylabel("交叉熵损失")
        axes[0].set_title("损失曲线")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].set_xlabel("训练轮次")
        axes[1].set_ylabel("准确率")
        axes[1].set_title("准确率曲线")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(out, dpi=150)
    finally:
        plt.close(fig)
    return out


def main() -> None:
    """CLI 入口：读取一组或多组训练日志并输出对比曲线图。"""
    ap = argparse.ArgumentParser(description="绘制 HAM10000 训练/验证曲线")
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/figures/train_curves.png")
    args = ap.parse_args()

    out = plot_training_curves(args.logs, args.names, args.out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
