"""测试 parquet_extract.py：多 parquet 流式还原、清单过滤、去重与目录穿越防护。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from parquet_extract import extract_image_bytes, extract_parquet_dataset, normalize_dx
from prepare import verify_no_lesion_leakage


def _jpeg_bytes(val: int = 100) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (24, 24), color=(val, 50, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def _write_shard(path: Path, rows: list[dict[str, object]]) -> None:
    image_type = pa.struct([("bytes", pa.binary())])
    table = pa.table(
        {
            "image": pa.array([{"bytes": r["bytes"]} for r in rows], type=image_type),
            "image_id": pa.array([r["image_id"] for r in rows], type=pa.string()),
            "lesion_id": pa.array([r["lesion_id"] for r in rows], type=pa.string()),
            "dx": pa.array([r["dx"] for r in rows], type=pa.string()),
        }
    )
    pq.write_table(table, path)


def test_normalize_dx_supports_official_and_case_variants() -> None:
    """支持官方英文全称、大小写变体及标准七类简写，拒绝未知诊断。"""
    assert normalize_dx("melanocytic_Nevi") == "nv"
    assert normalize_dx("melanocytic_nevi") == "nv"
    assert normalize_dx("melanoma") == "mel"
    assert normalize_dx("akiec") == "akiec"
    with pytest.raises(ValueError, match="未知 dx"):
        normalize_dx("unknown_lesion")
    with pytest.raises(ValueError, match="未知 dx"):
        normalize_dx(None)


def test_extract_image_bytes_validates_payload() -> None:
    """缺少有效字节内容时抛出清晰 ValueError。"""
    raw = b"\xff\xd8\xff"
    assert extract_image_bytes({"bytes": raw}, "ISIC_1") == raw
    assert extract_image_bytes(raw, "ISIC_1") == raw
    with pytest.raises(ValueError):
        extract_image_bytes({"bytes": b""}, "ISIC_1")
    with pytest.raises(ValueError):
        extract_image_bytes(None, "ISIC_1")


def test_extract_parquet_filters_official_dedups_and_zero_leakage(tmp_path: Path) -> None:
    """完整验证官方清单过滤、跨分片去重、图像落地与严格零病灶跨集泄漏。"""
    src_dir = tmp_path / "src_shards"
    src_dir.mkdir()
    jpg = _jpeg_bytes()

    dx_full = [
        "actinic_keratoses",
        "basal_cell_carcinoma",
        "benign_keratosis-like_lesions",
        "dermatofibroma",
        "melanoma",
        "melanocytic_Nevi",
        "vascular_lesions",
    ]
    shard1_rows: list[dict[str, object]] = []
    shard2_rows: list[dict[str, object]] = []
    official_ids: list[str] = []

    idx = 0
    for c_idx, dx_name in enumerate(dx_full):
        for lesion_num in range(6):
            lid = f"HAM_{c_idx}_{lesion_num}"
            for copy_k in range(2):
                iid = f"ISIC_{idx:06d}"
                idx += 1
                official_ids.append(iid)
                rec = {"bytes": jpg, "image_id": iid, "lesion_id": lid, "dx": dx_name}
                if copy_k == 0:
                    shard1_rows.append(rec)
                else:
                    shard2_rows.append(rec)

    # 在分片 2 中混入重复 image_id 与非官方清单噪声图像
    shard2_rows.append(shard1_rows[0])
    shard2_rows.append(
        {"bytes": jpg, "image_id": "ISIC_EXTRA_NOISE", "lesion_id": "HAM_EXTRA", "dx": "melanoma"}
    )

    _write_shard(src_dir / "part_00.parquet", shard1_rows)
    _write_shard(src_dir / "part_01.parquet", shard2_rows)

    meta_csv = tmp_path / "HAM10000_metadata.csv"
    meta_csv.write_text("image_id\n" + "\n".join(official_ids) + "\n", encoding="utf-8")

    out_dir = tmp_path / "ham_out"
    split_records, _ = extract_parquet_dataset(
        src_dir=src_dir,
        out_dir=out_dir,
        metadata=meta_csv,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        batch_size=16,
    )

    # 84 张官方图应全部保留，去重 + 过滤噪声成功
    assert len(split_records) == len(official_ids)
    assert "ISIC_EXTRA_NOISE" not in {r["image_id"] for r in split_records}
    assert not (out_dir / "images" / "ISIC_EXTRA_NOISE.jpg").exists()

    # 严格零病灶跨集泄漏
    verify_no_lesion_leakage(split_records)
    assert all((out_dir / "images" / f"{iid}.jpg").is_file() for iid in official_ids)


def test_extract_parquet_blocks_path_traversal_and_empty_dir(tmp_path: Path) -> None:
    """恶意 image_id 目录穿越与空分片目录均必须报错且不污染外部文件。"""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="parquet"):
        extract_parquet_dataset(empty_dir, tmp_path / "out_empty", metadata=None)

    evil_dir = tmp_path / "evil"
    evil_dir.mkdir()
    _write_shard(
        evil_dir / "bad.parquet",
        [
            {
                "bytes": _jpeg_bytes(),
                "image_id": "../pwned",
                "lesion_id": "HAM_001",
                "dx": "melanoma",
            }
        ],
    )
    out_dir = tmp_path / "out_evil"
    with pytest.raises(ValueError, match="目录穿越"):
        extract_parquet_dataset(evil_dir, out_dir, metadata=None)
    assert not (out_dir / "pwned.jpg").exists()
