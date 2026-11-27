"""从 HF 镜像的 parquet 分片还原 HAM10000 原始数据并按 lesion_id 分层划分。

镜像的 parquet 自带 train/val/test 划分是按图片随机分的，同一病灶的多张图
会同时落在训练和测试里（数据泄漏），因此这里忽略它：合并所有分片后，
按 lesion_id 分组重新做分层 train/val/test 划分。

用法：
    python src/parquet_extract.py --src-dir data_src --out-dir data/ham10000
输出：
    data/ham10000/images/*.jpg     # 10015 张原图（从 parquet 的 image.bytes 还原）
    data/ham10000/splits.csv       # image_id,lesion_id,dx,label,split
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from prepare import (
    CLASSES,
    resolve_safe_output_path,
    stratified_lesion_split,
    verify_no_lesion_leakage,
)

DX_MAP: dict[str, str] = {
    "actinic_keratoses": "akiec",
    "basal_cell_carcinoma": "bcc",
    "benign_keratosis-like_lesions": "bkl",
    "dermatofibroma": "df",
    "melanoma": "mel",
    "melanocytic_Nevi": "nv",
    "vascular_lesions": "vasc",
}
_DX_LOWER_MAP: dict[str, str] = {k.lower(): v for k, v in DX_MAP.items()}
_CLASS_SET: set[str] = set(CLASSES)


def normalize_dx(dx: str | None) -> str:
    """将 parquet 中的诊断名称标准化为 HAM10000 七类简写。

    Args:
        dx: 原始诊断标签（支持英文全称或标准七类代码）。

    Returns:
        标准类别代码（``akiec``/``bcc``/``bkl``/``df``/``mel``/``nv``/``vasc``）。

    Raises:
        ValueError: 标签为空或无法映射到合法七类。
    """
    if dx is None:
        raise ValueError("未知 dx: None")
    raw = str(dx).strip()
    if raw in DX_MAP:
        return DX_MAP[raw]
    low = raw.lower()
    if low in _DX_LOWER_MAP:
        return _DX_LOWER_MAP[low]
    if low in _CLASS_SET:
        return low
    raise ValueError(f"未知 dx: {dx}")


def extract_image_bytes(image_cell: object, image_id: str) -> bytes:
    """从 parquet 图像单元中安全提取二进制字节。"""
    if isinstance(image_cell, dict):
        data = image_cell.get("bytes")
        if isinstance(data, (bytes, bytearray)) and len(data) > 0:
            return bytes(data)
    elif isinstance(image_cell, (bytes, bytearray)) and len(image_cell) > 0:
        return bytes(image_cell)
    raise ValueError(f"图像 {image_id!r} 缺少有效的二进制字节内容")


def extract_parquet_dataset(
    src_dir: str | Path,
    out_dir: str | Path,
    metadata: str | Path | None = "data_src/HAM10000_metadata.csv",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    batch_size: int = 512,
) -> tuple[list[dict[str, str]], dict[str, Counter[str]]]:
    """流式读取 parquet 分片、过滤官方清单、还原图像并按病灶分层划分。

    Args:
        src_dir: 存放 ``*.parquet`` 分片的目录。
        out_dir: 输出目录（生成 ``images/`` 与 ``splits.csv``）。
        metadata: 可选官方 ``HAM10000_metadata.csv`` 路径；若存在则仅保留清单内的 ``image_id``。
        val_ratio: 验证集病灶比例。
        test_ratio: 测试集病灶比例。
        seed: 随机种子。
        batch_size: parquet 流式读取批大小，用于控制峰值内存。

    Returns:
        二元组 ``(split_records, split_counts)``。

    Raises:
        FileNotFoundError: ``src_dir`` 中未找到任何 ``.parquet`` 文件。
        ValueError: 遇到非法 ``image_id``（目录穿越）、未知 ``dx``、空数据或病灶诊断冲突。
    """
    src_path = Path(src_dir)
    shards = sorted(src_path.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"目录 {src_path} 中未找到任何 .parquet 分片文件")

    out = Path(out_dir)
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    official: set[str] = set()
    if metadata and Path(metadata).is_file():
        with open(metadata, encoding="utf-8-sig") as f:
            official = {r["image_id"].strip() for r in csv.DictReader(f) if r.get("image_id")}
        print(f"官方 metadata 清单 {len(official)} 张，将按它过滤")

    seen: set[str] = set()
    records: list[dict[str, str]] = []
    print(f"{len(shards)} 个 parquet 分片")

    required_cols = ["image", "image_id", "lesion_id", "dx"]
    for sp in shards:
        pq_file = pq.ParquetFile(sp)
        missing_cols = set(required_cols) - set(pq_file.schema_arrow.names)
        if missing_cols:
            raise ValueError(f"分片 {sp.name} 缺少必需列: {sorted(missing_cols)}")

        for batch in pq_file.iter_batches(batch_size=batch_size, columns=required_cols):
            imgs = batch.column("image").to_pylist()
            iids = batch.column("image_id").to_pylist()
            lids = batch.column("lesion_id").to_pylist()
            dxs = batch.column("dx").to_pylist()

            for img, iid_raw, lid_raw, dx in zip(imgs, iids, lids, dxs, strict=True):
                short = normalize_dx(dx)
                iid = str(iid_raw).strip() if iid_raw is not None else ""
                lid = str(lid_raw).strip() if lid_raw is not None else ""
                if not iid or not lid:
                    raise ValueError(f"分片 {sp.name} 存在空的 image_id 或 lesion_id")
                if official and iid not in official:
                    continue
                if iid in seen:  # 分片间可能有重复行
                    continue
                # 防范恶意 image_id 造成目录穿越
                dst = resolve_safe_output_path(img_out, f"{iid}.jpg")
                seen.add(iid)
                if not dst.exists():
                    dst.write_bytes(extract_image_bytes(img, iid))
                records.append({"image_id": iid, "lesion_id": lid, "dx": short})

            # 及时释放当前批次的大块图像字节引用
            del imgs, iids, lids, dxs, batch

        del pq_file
        print(f"  {sp.name}: 累计 {len(records)} 张")

    if not records:
        raise ValueError("过滤后未保留任何有效样本，请检查 parquet 分片与官方 metadata 是否匹配")

    lesion_split, lesion_dx = stratified_lesion_split(
        records, classes=CLASSES, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    split_records: list[dict[str, str]] = []
    for rec in records:
        lid = rec["lesion_id"]
        dx = rec["dx"]
        split_records.append(
            {
                "image_id": rec["image_id"],
                "lesion_id": lid,
                "dx": dx,
                "label": str(CLASSES.index(dx)),
                "split": lesion_split[lid],
            }
        )
    verify_no_lesion_leakage(split_records)

    with open(out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "lesion_id", "dx", "label", "split"])
        for rec in split_records:
            writer.writerow(
                [rec["image_id"], rec["lesion_id"], rec["dx"], rec["label"], rec["split"]]
            )

    cnt: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in split_records:
        cnt[rec["split"]][rec["dx"]] += 1

    print(f"\n{'class':>6} " + " ".join(f"{s:>6}" for s in ("train", "val", "test")))
    for cls in CLASSES:
        print(f"{cls:>6} " + " ".join(f"{cnt[s][cls]:>6}" for s in ("train", "val", "test")))
    total = sum(sum(v.values()) for v in cnt.values())
    print(f"合计 {total} 张图 / {len(lesion_dx)} 个病灶")
    return split_records, cnt


def main() -> None:
    """CLI 入口：从 parquet 分片还原 HAM10000 图像并按病灶划分。"""
    ap = argparse.ArgumentParser(description="从 parquet 分片还原 HAM10000 并按 lesion_id 分层划分")
    ap.add_argument("--src-dir", default="data_src")
    ap.add_argument("--out-dir", default="data/ham10000")
    ap.add_argument(
        "--metadata",
        default="data_src/HAM10000_metadata.csv",
        help="官方 metadata；parquet 里可能混入额外图像，只保留官方清单内的",
    )
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    extract_parquet_dataset(
        src_dir=args.src_dir,
        out_dir=args.out_dir,
        metadata=args.metadata,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
