"""测试 evaluate.py 与 make_test_results.py：指标计算边界、混淆矩阵与结果汇总。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dataset import CLASSES
from evaluate import (
    _compute_metrics_numpy,
    compute_classification_metrics,
    compute_confusion_matrix,
    plot_confusion_matrix,
    resolve_eval_save_dir,
)
from make_test_results import (
    denormalize_image,
    plot_prediction_examples,
    select_example_indices,
    write_predictions_csv,
)


def test_metrics_perfect_predictions() -> None:
    """全对预测时 Accuracy、Balanced Accuracy、Macro F1 与 Macro AUC 均应为 1.0。"""
    ys = np.arange(len(CLASSES), dtype=np.int64)
    ys = np.repeat(ys, 3)
    probs = np.full((len(ys), len(CLASSES)), 0.01, dtype=np.float64)
    for idx, label in enumerate(ys):
        probs[idx, label] = 0.94

    m = compute_classification_metrics(ys, probs, classes=CLASSES)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["balanced_acc"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["macro_auc"] == pytest.approx(1.0)
    assert m["cm_norm"].shape == (7, 7)
    assert np.allclose(m["cm_norm"], np.eye(7))


def test_metrics_majority_collapse_exposes_low_balanced_acc() -> None:
    """模拟全猜 nv（大类塌缩）：Accuracy 虚高但 Balanced Accuracy 仅为 1/7。"""
    ys = np.array([0, 1, 2, 3, 4, 6] + [5] * 94, dtype=np.int64)
    probs = np.zeros((len(ys), 7), dtype=np.float64)
    probs[:, 5] = 0.99
    probs[:, 0] = 0.01

    m = compute_classification_metrics(ys, probs, classes=CLASSES)
    assert m["accuracy"] == pytest.approx(0.94)
    assert m["balanced_acc"] == pytest.approx(1.0 / 7.0)


def test_metrics_missing_classes_boundary_safe() -> None:
    """关键边界：测试集缺失多个稀有类时，混淆矩阵仍恒为 7×7，不越界且 AUC 安全返回 nan。"""
    ys = np.array([0, 0, 4, 4, 5, 5], dtype=np.int64)
    probs = np.array(
        [
            [0.8, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
            [0.3, 0.0, 0.0, 0.0, 0.6, 0.1, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.2, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    m = compute_classification_metrics(ys, probs, classes=CLASSES)
    assert m["cm_raw"].shape == (7, 7)
    assert m["cm_norm"].shape == (7, 7)
    assert np.all(np.isfinite(m["cm_norm"]))
    # 缺失类别所在行应为全 0，而非 NaN
    assert np.all(m["cm_norm"][1] == 0.0)
    assert np.all(m["cm_norm"][6] == 0.0)
    assert np.isnan(m["macro_auc"])
    assert m["balanced_acc"] == pytest.approx((0.5 + 0.5 + 1.0) / 3.0)
    for cls in CLASSES:
        assert cls in m["report"]


def test_metrics_numpy_matches_sklearn_when_available() -> None:
    """当环境安装 scikit-learn 时，交叉验证纯 NumPy 引擎与 sklearn 数值完全一致。"""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(123)
    ys = rng.integers(0, len(CLASSES), size=140)
    raw = rng.random((140, len(CLASSES)))
    probs = raw / raw.sum(axis=1, keepdims=True)

    m_np = _compute_metrics_numpy(ys, probs, classes=CLASSES)
    m_sk = compute_classification_metrics(ys, probs, classes=CLASSES, use_sklearn=True)

    assert m_np["accuracy"] == pytest.approx(m_sk["accuracy"], abs=1e-10)
    assert m_np["balanced_acc"] == pytest.approx(m_sk["balanced_acc"], abs=1e-10)
    assert m_np["macro_f1"] == pytest.approx(m_sk["macro_f1"], abs=1e-10)
    assert m_np["macro_auc"] == pytest.approx(m_sk["macro_auc"], abs=1e-10)
    assert np.array_equal(m_np["cm_raw"], m_sk["cm_raw"])
    assert np.allclose(m_np["cm_norm"], m_sk["cm_norm"], atol=1e-10)


def test_metrics_invalid_inputs_raise() -> None:
    """空输入与形状不匹配必须抛出 ValueError。"""
    with pytest.raises(ValueError):
        compute_classification_metrics(np.array([], dtype=np.int64), np.zeros((0, 7)))
    with pytest.raises(ValueError):
        compute_classification_metrics(np.array([0, 1]), np.zeros((2, 3)), classes=CLASSES)
    with pytest.raises(ValueError):
        compute_confusion_matrix(np.array([0]), np.array([0]), normalize="invalid")


def test_plot_confusion_matrix_closes_figure(tmp_path: Path) -> None:
    """混淆矩阵绘图后必须生成 PNG 且关闭 Matplotlib figure 避免内存泄漏。"""
    plt.close("all")
    cm = np.eye(len(CLASSES), dtype=np.float64)
    out = plot_confusion_matrix(cm, CLASSES, 0.726, tmp_path / "cm.png")
    assert out.is_file()
    assert out.stat().st_size > 0
    assert plt.get_fignums() == []


def test_resolve_eval_save_dir_handles_shallow_path() -> None:
    """传入单层文件名如 best.pth 时不会触发 parents[1] 越界异常。"""
    resolved = resolve_eval_save_dir("best.pth")
    assert resolved.name == "eval"
    explicit = resolve_eval_save_dir("best.pth", save_dir="custom/eval")
    assert explicit == Path("custom/eval")


def test_make_test_results_csv_and_small_sample_selection(tmp_path: Path) -> None:
    """验证逐图预测明细 CSV 写出、反归一化与极小样本下不崩溃抽样。"""
    keys = ["ISIC_1", "ISIC_2", "ISIC_3"]
    lesion_of = {"ISIC_1": "HAM_1", "ISIC_2": "HAM_2", "ISIC_3": "HAM_3"}
    ys = np.array([4, 5, 1], dtype=np.int64)  # mel, nv, bcc
    preds = np.array([5, 5, 2], dtype=np.int64)  # 错, 对, 错
    conf = np.array([0.82, 0.99, 0.61], dtype=np.float64)

    csv_path = tmp_path / "test_predictions.csv"
    acc = write_predictions_csv(csv_path, keys, lesion_of, ys, preds, conf)
    assert acc == pytest.approx(1.0 / 3.0)

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["真实类"] == "mel"
    assert rows[0]["预测类"] == "nv"
    assert rows[0]["是否正确"] == "错误"
    assert rows[1]["是否正确"] == "正确"

    # 边界测试：仅 1 张判对、2 张高危判错、0 张非高危判错时绝不崩溃
    right_pick, wrong_pick = select_example_indices(ys, preds, classes=CLASSES, seed=42)
    assert right_pick == [1]
    assert set(wrong_pick) == {0, 2}

    # 边界测试：全部判对（0 张判错）时也不崩溃
    right_all, wrong_none = select_example_indices(ys, ys, classes=CLASSES, seed=42)
    assert len(right_all) == 3
    assert wrong_none == []

    # 反归一化与绘图释放 figure 检查
    dummy = denormalize_image(np.zeros((16, 16, 3), dtype=np.float64))
    assert float(dummy.min()) >= 0.0 and float(dummy.max()) <= 1.0

    plt.close("all")
    fig_path = plot_prediction_examples(
        image_loader=lambda _i: dummy,
        ys=ys,
        preds=preds,
        conf=conf,
        right_pick=right_pick,
        wrong_pick=wrong_pick,
        out_path=tmp_path / "examples.png",
    )
    assert fig_path.is_file()
    assert plt.get_fignums() == []


def test_real_test_predictions_csv_integrity_readonly() -> None:
    """只读核验仓库内真实实验产物 results/test_predictions.csv 的完整性与准确率。"""
    csv_path = Path(__file__).resolve().parent.parent / "results" / "test_predictions.csv"
    if not csv_path.is_file():
        pytest.skip("仓库缺少 results/test_predictions.csv")

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1001
    correct = sum(1 for r in rows if r["是否正确"] == "正确")
    assert correct == 828
    assert all(r["真实类"] in CLASSES and r["预测类"] in CLASSES for r in rows)
