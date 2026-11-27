"""测试 plot_curves.py、eda.py、cn_font.py 与 train.py 辅助函数。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cn_font import configure_cn_font
from dataset import CLASSES
from eda import plot_class_distribution, plot_class_examples
from plot_curves import plot_training_curves, read_log
from train import resolve_out_dir, set_seed, validate_train_args


def test_read_log_supports_single_and_multi_epoch_and_rejects_bad_csv(tmp_path: Path) -> None:
    """单行 1-epoch 日志必须正常返回一维数组，空文件与缺列必须抛 ValueError。"""
    single = tmp_path / "single.csv"
    single.write_text(
        "epoch,train_loss,train_acc,val_loss,val_acc\n1,1.2500,0.6100,1.1000,0.6500\n",
        encoding="utf-8",
    )
    ep, tl, ta, vl, va = read_log(single)
    assert ep.shape == (1,)
    assert tl[0] == pytest.approx(1.25)
    assert va[0] == pytest.approx(0.65)

    empty = tmp_path / "empty.csv"
    empty.write_text("epoch,train_loss,train_acc,val_loss,val_acc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="为空"):
        read_log(empty)

    bad_cols = tmp_path / "bad.csv"
    bad_cols.write_text("epoch,train_loss\n1,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必需列"):
        read_log(bad_cols)


def test_plot_training_curves_closes_figures_and_checks_length(tmp_path: Path) -> None:
    """验证训练曲线生成、figure 释放及 --logs/--names 数量一致性校验。"""
    log1 = tmp_path / "l1.csv"
    log1.write_text(
        "epoch,train_loss,train_acc,val_loss,val_acc\n"
        "1,1.5,0.50,1.4,0.52\n"
        "2,1.1,0.68,1.2,0.65\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="数量不一致"):
        plot_training_curves([log1], ["a", "b"], tmp_path / "mismatch.png")

    plt.close("all")
    out = plot_training_curves([log1], ["ResNet18 (scratch)"], tmp_path / "curves.png")
    assert out.is_file()
    assert out.stat().st_size > 0
    assert plt.get_fignums() == []


def test_eda_plots_without_nv_class_and_closes_figures(tmp_path: Path) -> None:
    """当数据子集不含 nv 类别时，EDA 柱状图仍应自适应坐标轴上限而不崩溃。"""
    rows = [
        {"image_id": "ISIC_1", "lesion_id": "HAM_1", "dx": "mel", "label": "4", "split": "train"},
        {"image_id": "ISIC_2", "lesion_id": "HAM_2", "dx": "mel", "label": "4", "split": "train"},
        {"image_id": "ISIC_3", "lesion_id": "HAM_3", "dx": "bcc", "label": "1", "split": "train"},
    ]
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    sample_img = np.full((32, 32, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", sample_img)
    assert ok
    for row in rows:
        (img_dir / f"{row['image_id']}.jpg").write_bytes(encoded.tobytes())

    plt.close("all")
    dist_png = plot_class_distribution(rows, tmp_path / "dist.png")
    ex_png = plot_class_examples(rows, img_dir, tmp_path / "examples.png", split="train")
    assert dist_png.is_file() and dist_png.stat().st_size > 0
    assert ex_png.is_file() and ex_png.stat().st_size > 0
    assert plt.get_fignums() == []

    with pytest.raises(ValueError, match="为空"):
        plot_class_distribution([], tmp_path / "empty.png")


def test_cn_font_configuration_fallback(tmp_path: Path) -> None:
    """字体文件不存在时应安全配置回退列表并禁用 unicode_minus。"""
    fonts = configure_cn_font(tmp_path / "missing_simhei.ttf")
    assert len(fonts) >= 2
    assert matplotlib.rcParams["axes.unicode_minus"] is False
    assert matplotlib.rcParams["font.sans-serif"] == fonts


def test_train_helpers_and_arg_validation() -> None:
    """验证训练输出目录推导、随机种子固定与超参数边界校验。"""
    set_seed(42)
    assert resolve_out_dir("resnet18_scratch", weighted=False) == Path("outputs/scratch")
    assert resolve_out_dir("resnet18_pretrained", weighted=True) == Path(
        "outputs/pretrained_weighted"
    )
    assert resolve_out_dir("resnet18_scratch", weighted=False, out_dir="custom/out") == Path(
        "custom/out"
    )

    valid_args = argparse.Namespace(epochs=30, batch_size=64, lr=1e-3, size=224, workers=4)
    validate_train_args(valid_args)

    for bad_kwargs in (
        {"epochs": 0},
        {"batch_size": 0},
        {"lr": -1e-3},
        {"size": 0},
        {"workers": -1},
    ):
        merged = {**vars(valid_args), **bad_kwargs}
        with pytest.raises(ValueError):
            validate_train_args(argparse.Namespace(**merged))

    assert len(CLASSES) == 7
