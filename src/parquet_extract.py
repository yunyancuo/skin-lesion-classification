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
import argparse
import csv
import glob
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DX_MAP = {
    "actinic_keratoses": "akiec", "basal_cell_carcinoma": "bcc",
    "benign_keratosis-like_lesions": "bkl", "dermatofibroma": "df",
    "melanoma": "mel", "melanocytic_Nevi": "nv", "vascular_lesions": "vasc",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="data_src")
    ap.add_argument("--out-dir", default="data/ham10000")
    ap.add_argument("--metadata", default="data_src/HAM10000_metadata.csv",
                    help="官方 metadata；parquet 里可能混入额外图像，只保留官方清单内的")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    import csv as _csv
    official = set()
    if args.metadata and Path(args.metadata).exists():
        with open(args.metadata, encoding="utf-8") as f:
            official = {r["image_id"] for r in _csv.DictReader(f)}
        print(f"官方 metadata 清单 {len(official)} 张，将按它过滤")

    seen = set()
    records = []  # (image_id, lesion_id, dx_short)
    shards = sorted(glob.glob(str(Path(args.src_dir) / "*.parquet")))
    print(f"{len(shards)} 个 parquet 分片")
    for sp in shards:
        tbl = pq.read_table(sp, columns=["image", "image_id", "lesion_id", "dx"])
        imgs = tbl.column("image").to_pylist()
        iids = tbl.column("image_id").to_pylist()
        lids = tbl.column("lesion_id").to_pylist()
        dxs = tbl.column("dx").to_pylist()
        for img, iid, lid, dx in zip(imgs, iids, lids, dxs):
            short = DX_MAP.get(dx)
            if short is None:
                raise ValueError(f"未知 dx: {dx}")
            if official and iid not in official:
                continue
            if iid in seen:  # 分片间可能有重复行
                continue
            seen.add(iid)
            dst = img_out / f"{iid}.jpg"
            if not dst.exists():
                dst.write_bytes(img["bytes"])
            records.append((iid, lid, short))
        print(f"  {Path(sp).name}: 累计 {len(records)} 张")

    by_lesion = defaultdict(set)
    for iid, lid, dx in records:
        by_lesion[lid].add(dx)
    assert all(len(v) == 1 for v in by_lesion.values()), "同一 lesion_id 出现多种诊断？"

    lesion_dx = {lid: next(iter(v)) for lid, v in by_lesion.items()}
    lesion_split = {}
    for c in CLASSES:
        lesions = sorted(lid for lid, dx in lesion_dx.items() if dx == c)
        rng.shuffle(lesions)
        n = len(lesions)
        n_val, n_test = round(n * args.val_ratio), round(n * args.test_ratio)
        n_train = n - n_val - n_test
        for i, lid in enumerate(lesions):
            lesion_split[lid] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    with open(out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "lesion_id", "dx", "label", "split"])
        for iid, lid, dx in records:
            w.writerow([iid, lid, dx, CLASSES.index(dx), lesion_split[lid]])

    cnt = defaultdict(Counter)
    for _, lid, dx in records:
        cnt[lesion_split[lid]][dx] += 1
    print(f"\n{'class':>6} " + " ".join(f"{s:>6}" for s in ("train", "val", "test")))
    for c in CLASSES:
        print(f"{c:>6} " + " ".join(f"{cnt[s][c]:>6}" for s in ("train", "val", "test")))
    total = sum(sum(v.values()) for v in cnt.values())
    print(f"合计 {total} 张图 / {len(by_lesion)} 个病灶")


if __name__ == "__main__":
    main()
