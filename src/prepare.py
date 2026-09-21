"""HAM10000 数据准备：解包 Kaggle zip，按 lesion_id 分层划分 train/val/test。

同一病灶常有多张图片（10015 张图 / 7470 个病灶），划分必须按病灶分组，
否则同一病灶的图同时出现在训练和测试集里，会造成数据泄漏、指标虚高。

用法：
    python src/prepare.py --zip data/ham10000.zip --out-dir data/ham10000
输出：
    data/ham10000/images/*.jpg          # 全部图像（软链接或复制）
    data/ham10000/splits.csv            # image_id,lesion_id,dx,split
"""
import argparse
import csv
import random
import zipfile
from collections import defaultdict
from pathlib import Path

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def find_metadata(zip_names):
    cands = [n for n in zip_names if n.endswith("HAM10000_metadata.csv")]
    if not cands:
        raise FileNotFoundError("zip 里没找到 HAM10000_metadata.csv")
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/ham10000.zip")
    ap.add_argument("--out-dir", default="data/ham10000")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(args.zip)
    names = zf.namelist()
    meta_name = find_metadata(names)
    reader = csv.DictReader(io_text(zf.read(meta_name)))
    rows = list(reader)
    print(f"metadata {len(rows)} 行")

    # lesion_id -> [(image_id, dx), ...]
    by_lesion = defaultdict(list)
    for r in rows:
        by_lesion[r["lesion_id"]].append((r["image_id"], r["dx"]))

    # 按类别分层：每个类里按病灶分组轮流分配到 train/val/test，比例近似
    lesion_split = {}
    for c in CLASSES:
        lesions = sorted({lid for lid, items in by_lesion.items() if items[0][1] == c})
        rng.shuffle(lesions)
        n = len(lesions)
        n_val, n_test = round(n * args.val_ratio), round(n * args.test_ratio)
        n_train = n - n_val - n_test
        for i, lid in enumerate(lesions):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            lesion_split[lid] = split

    with open(out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "lesion_id", "dx", "label", "split"])
        for r in rows:
            lid, iid, dx = r["lesion_id"], r["image_id"], r["dx"]
            w.writerow([iid, lid, dx, CLASSES.index(dx), lesion_split[lid]])

    cnt = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cnt[lesion_split[r["lesion_id"]]][r["dx"]] += 1
    print(f"{'class':>6} " + " ".join(f"{s:>6}" for s in ("train", "val", "test")))
    for c in CLASSES:
        print(f"{c:>6} " + " ".join(f"{cnt[s][c]:>6}" for s in ("train", "val", "test")))

    # 解包图像（部分镜像 zip 里目录名不同，统一平铺到 images/）
    extracted = 0
    for n in names:
        if n.lower().endswith((".jpg", ".jpeg", ".png")) and ("part" in n.lower() or "images" in n.lower()):
            dst = img_out / Path(n).name
            if not dst.exists():
                dst.write_bytes(zf.read(n))
            extracted += 1
    print(f"解包图像 {extracted} 张 -> {img_out}")
    missing = [r["image_id"] for r in rows if not (img_out / f"{r['image_id']}.jpg").exists()]
    print(f"缺失图像: {len(missing)}")


def io_text(b):
    import io
    return io.TextIOWrapper(io.BytesIO(b), encoding="utf-8")


if __name__ == "__main__":
    main()
