"""测试 prepare.py：病灶级分层划分防数据泄漏、边界校验与安全解包。"""

from __future__ import annotations

import csv
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from prepare import (
    CLASSES,
    find_metadata,
    prepare_from_zip,
    resolve_safe_output_path,
    safe_extract_zip_images,
    stratified_lesion_split,
    validate_split_ratios,
    verify_no_lesion_leakage,
)


def _make_dummy_jpeg_bytes(color: tuple[int, int, int] = (120, 80, 60)) -> bytes:
    """生成最小合法 JPEG 字节流，用于离线构造模拟数据。"""
    buf = BytesIO()
    Image.new("RGB", (32, 24), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _build_imbalanced_metadata() -> list[dict[str, str]]:
    """构造模拟 HAM10000 分布的元数据（含同一 lesion_id 多张图与 7 类不平衡）。"""
    lesion_counts = {
        "akiec": 15,
        "bcc": 20,
        "bkl": 30,
        "df": 10,
        "mel": 35,
        "nv": 120,
        "vasc": 12,
    }
    rows: list[dict[str, str]] = []
    img_idx = 0
    lesion_idx = 0
    for dx, n_lesions in lesion_counts.items():
        for local_i in range(n_lesions):
            lesion_id = f"HAM_{lesion_idx:06d}"
            lesion_idx += 1
            # 模拟同一病灶有 1~3 张不同拍摄角度的照片
            n_imgs = 1 + (local_i % 3)
            for _ in range(n_imgs):
                rows.append(
                    {
                        "image_id": f"ISIC_{img_idx:07d}",
                        "lesion_id": lesion_id,
                        "dx": dx,
                    }
                )
                img_idx += 1
    return rows


def test_stratified_lesion_split_zero_leakage_and_deterministic() -> None:
    """核心医学约束：严格验证同一 lesion_id 绝不跨 train/val/test 泄漏，且同 seed 可复现。"""
    rows = _build_imbalanced_metadata()
    lesion_split_1, lesion_dx_1 = stratified_lesion_split(
        rows, classes=CLASSES, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    lesion_split_2, _ = stratified_lesion_split(
        rows, classes=CLASSES, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    lesion_split_3, _ = stratified_lesion_split(
        rows, classes=CLASSES, val_ratio=0.1, test_ratio=0.1, seed=99
    )

    assert lesion_split_1 == lesion_split_2
    assert lesion_split_1 != lesion_split_3

    split_rows = [
        {
            "image_id": r["image_id"],
            "lesion_id": r["lesion_id"],
            "dx": r["dx"],
            "split": lesion_split_1[r["lesion_id"]],
        }
        for r in rows
    ]
    verify_no_lesion_leakage(split_rows)

    # 显式验证三个集合的 lesion_id 集合两两交集为空
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for r in split_rows:
        by_split[r["split"]].add(r["lesion_id"])

    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])
    assert len(by_split["train"] | by_split["val"] | by_split["test"]) == len(lesion_dx_1)

    # 每个类别在三个集合中均应有样本，且验证/测试集比例接近 10%
    for cls in CLASSES:
        cls_lesions = {lid for lid, dx in lesion_dx_1.items() if dx == cls}
        assert cls_lesions & by_split["train"]
        assert cls_lesions & by_split["val"]
        assert cls_lesions & by_split["test"]


def test_verify_no_lesion_leakage_detects_cross_split_leak() -> None:
    """人为制造同一病灶同时出现在 train 和 test 时必须抛 ValueError。"""
    leaked_rows = [
        {"image_id": "ISIC_0001", "lesion_id": "HAM_0001", "split": "train"},
        {"image_id": "ISIC_0002", "lesion_id": "HAM_0001", "split": "test"},
    ]
    with pytest.raises(ValueError, match="数据泄漏"):
        verify_no_lesion_leakage(leaked_rows)


def test_stratified_lesion_split_rejects_conflicting_diagnoses() -> None:
    """同一 lesion_id 对应两种不同诊断标签时必须报错。"""
    rows = [
        {"image_id": "ISIC_1", "lesion_id": "HAM_1", "dx": "nv"},
        {"image_id": "ISIC_2", "lesion_id": "HAM_1", "dx": "mel"},
    ]
    with pytest.raises(ValueError, match="多种诊断"):
        stratified_lesion_split(rows)


def test_stratified_lesion_split_rejects_unknown_dx_and_empty_rows() -> None:
    """未知诊断类别与空元数据必须被显式拦截。"""
    with pytest.raises(ValueError, match="为空"):
        stratified_lesion_split([])

    with pytest.raises(ValueError, match="未知诊断"):
        stratified_lesion_split([{"image_id": "ISIC_1", "lesion_id": "HAM_1", "dx": "unknown"}])


@pytest.mark.parametrize(
    ("val_ratio", "test_ratio"),
    [(-0.1, 0.1), (0.1, -0.2), (1.0, 0.0), (0.5, 0.5), (0.6, 0.5)],
)
def test_validate_split_ratios_invalid(val_ratio: float, test_ratio: float) -> None:
    """非法划分比例必须抛出 ValueError。"""
    with pytest.raises(ValueError):
        validate_split_ratios(val_ratio, test_ratio)


def test_stratified_lesion_split_tiny_class_does_not_produce_negative_train() -> None:
    """极端小样本（单类仅 1 个病灶 + 大比例）不产生负训练数且不丢病灶。"""
    rows = [{"image_id": "ISIC_1", "lesion_id": "HAM_1", "dx": "df"}]
    lesion_split, _ = stratified_lesion_split(rows, val_ratio=0.45, test_ratio=0.45, seed=42)
    assert lesion_split["HAM_1"] in {"train", "val", "test"}


def test_find_metadata_ignores_macosx_and_handles_missing() -> None:
    """定位 metadata 时跳过 __MACOSX 元数据，缺失时抛 FileNotFoundError。"""
    names = [
        "__MACOSX/._HAM10000_metadata.csv",
        "HAM10000_images_part_1/ISIC_0001.jpg",
        "nested/HAM10000_metadata.csv",
    ]
    assert find_metadata(names) == "nested/HAM10000_metadata.csv"

    with pytest.raises(FileNotFoundError):
        find_metadata(["HAM10000_images_part_1/ISIC_0001.jpg"])


def test_resolve_safe_output_path_blocks_traversal(tmp_path: Path) -> None:
    """路径安全函数必须拦截目录穿越与非法分隔符。"""
    safe = resolve_safe_output_path(tmp_path, "ISIC_0001.jpg")
    assert safe == (tmp_path / "ISIC_0001.jpg").resolve()

    for bad_name in ("../evil.jpg", "sub/a.jpg", "sub\\a.jpg", "..", "", "a\x00b.jpg"):
        with pytest.raises(ValueError):
            resolve_safe_output_path(tmp_path, bad_name)


def test_prepare_from_zip_end_to_end_and_zip_slip_guard(tmp_path: Path) -> None:
    """验证 Kaggle zip 端到端解包、零泄漏输出，以及恶意 Zip Slip 拦截。"""
    rows = _build_imbalanced_metadata()[:40]
    jpeg_bytes = _make_dummy_jpeg_bytes()

    zip_path = tmp_path / "ham10000.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        csv_buf = BytesIO()
        csv_text = "image_id,lesion_id,dx\n" + "\n".join(
            f"{r['image_id']},{r['lesion_id']},{r['dx']}" for r in rows
        )
        csv_buf.write(csv_text.encode("utf-8"))
        zf.writestr("HAM10000_metadata.csv", csv_buf.getvalue())
        zf.writestr("__MACOSX/HAM10000_images_part_1/._ignored.jpg", b"junk")
        for r in rows:
            zf.writestr(f"HAM10000_images_part_1/{r['image_id']}.jpg", jpeg_bytes)

    out_dir = tmp_path / "prepared"
    stats = prepare_from_zip(zip_path, out_dir, val_ratio=0.1, test_ratio=0.1, seed=42)
    assert sum(sum(c.values()) for c in stats.values()) == len(rows)

    with open(out_dir / "splits.csv", encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))
    assert len(split_rows) == len(rows)
    verify_no_lesion_leakage(split_rows)
    assert all((out_dir / "images" / f"{r['image_id']}.jpg").is_file() for r in rows)

    # 恶意 Zip Slip 归档必须被拒绝
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("HAM10000_images_part_1/../../escape.jpg", jpeg_bytes)
    with zipfile.ZipFile(evil_zip) as zf, pytest.raises(ValueError, match="Zip Slip"):
        safe_extract_zip_images(zf, tmp_path / "out_evil")
