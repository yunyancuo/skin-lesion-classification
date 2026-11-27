"""HAM10000 数据准备：解包 Kaggle zip，按 lesion_id 分层划分 train/val/test。

同一病灶常有多张图片（10015 张图 / 7470 个病灶），划分必须按病灶分组，
否则同一病灶的图同时出现在训练和测试集里，会造成数据泄漏、指标虚高。

用法：
    python src/prepare.py --zip data/ham10000.zip --out-dir data/ham10000
输出：
    data/ham10000/images/*.jpg          # 全部图像（统一平铺到 images/）
    data/ham10000/splits.csv            # image_id,lesion_id,dx,label,split
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

CLASSES: list[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def find_metadata(zip_names: Sequence[str]) -> str:
    """在 zip 条目列表中定位 ``HAM10000_metadata.csv``。

    Args:
        zip_names: zip 归档内成员路径列表。

    Returns:
        匹配到的元数据文件成员名。

    Raises:
        FileNotFoundError: 归档内未找到 ``HAM10000_metadata.csv``。
    """
    cands = [
        name
        for name in zip_names
        if not name.startswith("__MACOSX/") and name.endswith("HAM10000_metadata.csv")
    ]
    if not cands:
        raise FileNotFoundError("zip 里没找到 HAM10000_metadata.csv")
    return cands[0]


def validate_split_ratios(val_ratio: float, test_ratio: float) -> None:
    """校验验证集与测试集划分比例是否合法。"""
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio 必须在 [0, 1) 范围内，当前为: {val_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio 必须在 [0, 1) 范围内，当前为: {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio 必须小于 1.0，当前为: {val_ratio + test_ratio:.4f}"
        )


def resolve_safe_output_path(base_dir: Path, filename: str) -> Path:
    """校验文件名安全性并返回限定在 ``base_dir`` 内的目标路径，防止目录穿越。

    Args:
        base_dir: 允许写入的目标根目录。
        filename: 纯文件名（不得包含路径分隔符、``..`` 或空字符）。

    Returns:
        经过校验的目标文件路径。

    Raises:
        ValueError: 文件名非法或解析后越出 ``base_dir``。
    """
    if (
        not filename
        or "\x00" in filename
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
        or ".." in Path(filename).parts
    ):
        raise ValueError(f"检测到非法文件名或潜在目录穿越: {filename!r}")

    base_resolved = base_dir.resolve()
    target = (base_resolved / filename).resolve()
    if target.parent != base_resolved:
        raise ValueError(f"目标路径越出允许目录: {filename!r}")
    return target


def stratified_lesion_split(
    rows: Sequence[Mapping[str, str]],
    classes: Sequence[str] = CLASSES,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[dict[str, str], dict[str, str]]:
    """按病灶（``lesion_id``）分组、按类别分层的 train/val/test 划分。

    核心保证：
    1. 同一 ``lesion_id`` 必须对应唯一诊断类别 ``dx``，否则报错；
    2. 同一 ``lesion_id`` 的全部图像只能落入同一个 split，严格零跨集泄漏。

    Args:
        rows: 包含 ``lesion_id`` 和 ``dx`` 字段的元数据记录序列。
        classes: 合法类别名称有序列表。
        val_ratio: 验证集病灶比例。
        test_ratio: 测试集病灶比例。
        seed: 随机种子。

    Returns:
        二元组 ``(lesion_split, lesion_dx)``，分别为 ``lesion_id -> split`` 与 ``lesion_id -> dx``。

    Raises:
        ValueError: 输入为空、比例非法、存在未知诊断或同一病灶对应多种诊断。
    """
    validate_split_ratios(val_ratio, test_ratio)
    if not rows:
        raise ValueError("metadata 为空，无法执行数据划分")

    class_set = set(classes)
    by_lesion: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        lid = str(row.get("lesion_id", "")).strip()
        dx = str(row.get("dx", "")).strip()
        if not lid:
            raise ValueError("metadata 中存在空的 lesion_id")
        if dx not in class_set:
            raise ValueError(f"未知诊断类别 dx: {dx!r}")
        by_lesion[lid].add(dx)

    conflicts = {lid: sorted(dxs) for lid, dxs in by_lesion.items() if len(dxs) > 1}
    if conflicts:
        sample_conflict = next(iter(conflicts.items()))
        raise ValueError(
            f"同一 lesion_id 出现多种诊断标签（共 {len(conflicts)} 个冲突），"
            f"例如: {sample_conflict}"
        )

    lesion_dx = {lid: next(iter(dxs)) for lid, dxs in by_lesion.items()}
    rng = random.Random(seed)
    lesion_split: dict[str, str] = {}

    for cls in classes:
        lesions = sorted(lid for lid, dx in lesion_dx.items() if dx == cls)
        rng.shuffle(lesions)
        n = len(lesions)
        n_val = round(n * val_ratio)
        n_test = round(n * test_ratio)
        # 极端小样本下防止 n_val + n_test 超出总数导致负数 n_train
        while n_val + n_test > n:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break
        n_train = n - n_val - n_test

        for idx, lid in enumerate(lesions):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            lesion_split[lid] = split

    return lesion_split, lesion_dx


def verify_no_lesion_leakage(split_rows: Iterable[Mapping[str, str]]) -> None:
    """校验划分结果中不存在任何 ``lesion_id`` 跨 split 泄漏。

    Args:
        split_rows: 包含 ``lesion_id`` 与 ``split`` 字段的记录迭代器。

    Raises:
        ValueError: 若发现任一 ``lesion_id`` 同时出现在多个 split 中。
    """
    lesion_splits: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        lid = str(row["lesion_id"]).strip()
        split = str(row["split"]).strip()
        lesion_splits[lid].add(split)

    leaked = {lid: sorted(splits) for lid, splits in lesion_splits.items() if len(splits) > 1}
    if leaked:
        preview = list(leaked.items())[:5]
        raise ValueError(
            f"检测到病灶级数据泄漏：共 {len(leaked)} 个 lesion_id 跨 split 出现，示例: {preview}"
        )


def io_text(b: bytes) -> io.TextIOWrapper:
    """将字节流包装为兼容 UTF-8 BOM 的文本流。"""
    return io.TextIOWrapper(io.BytesIO(b), encoding="utf-8-sig")


def safe_extract_zip_images(zf: zipfile.ZipFile, img_out: Path) -> int:
    """从 Kaggle HAM10000 zip 安全解包图像文件并平铺至 ``img_out``。

    会自动跳过目录条目与 macOS ``__MACOSX`` 元数据文件，并拦截路径穿越（Zip Slip）。

    Args:
        zf: 已打开的 ``ZipFile`` 对象。
        img_out: 图像输出目录。

    Returns:
        匹配并处理（含已存在跳过）的图像数量。

    Raises:
        ValueError: 遇到包含路径穿越片段的恶意 zip 成员名。
    """
    img_out.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for info in zf.infolist():
        name = info.filename
        if info.is_dir() or name.startswith("__MACOSX/"):
            continue
        low = name.lower()
        if not (low.endswith(IMAGE_SUFFIXES) and ("part" in low or "images" in low)):
            continue
        # 显式拦截压缩包路径中的目录穿越片段（同时兼容 POSIX 与 Windows 分隔符）
        norm_parts = name.replace("\\", "/").split("/")
        if ".." in norm_parts or name.startswith(("/", "\\")):
            raise ValueError(f"检测到非法 zip 条目路径（疑似 Zip Slip）: {name!r}")
        filename = norm_parts[-1]
        dst = resolve_safe_output_path(img_out, filename)
        if not dst.exists():
            dst.write_bytes(zf.read(info))
        extracted += 1
    return extracted


def prepare_from_zip(
    zip_path: str | Path,
    out_dir: str | Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """执行完整的 Kaggle zip 解包、病灶级分层划分与防泄漏校验。

    Args:
        zip_path: HAM10000 zip 压缩包路径。
        out_dir: 输出目录（生成 ``images/`` 与 ``splits.csv``）。
        val_ratio: 验证集比例。
        test_ratio: 测试集比例。
        seed: 随机种子。

    Returns:
        按 ``split -> dx -> 图像数`` 统计的字典。
    """
    out = Path(out_dir)
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        meta_name = find_metadata(names)
        with io_text(zf.read(meta_name)) as text_stream:
            rows = list(csv.DictReader(text_stream))
        print(f"metadata {len(rows)} 行")

        lesion_split, _ = stratified_lesion_split(
            rows, classes=CLASSES, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )

        split_records = [
            {
                "image_id": r["image_id"],
                "lesion_id": r["lesion_id"],
                "dx": r["dx"],
                "label": str(CLASSES.index(r["dx"])),
                "split": lesion_split[r["lesion_id"]],
            }
            for r in rows
        ]
        verify_no_lesion_leakage(split_records)

        with open(out / "splits.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "lesion_id", "dx", "label", "split"])
            for rec in split_records:
                writer.writerow(
                    [rec["image_id"], rec["lesion_id"], rec["dx"], rec["label"], rec["split"]]
                )

        cnt: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for rec in split_records:
            cnt[rec["split"]][rec["dx"]] += 1

        print(f"{'class':>6} " + " ".join(f"{s:>6}" for s in ("train", "val", "test")))
        for cls in CLASSES:
            print(f"{cls:>6} " + " ".join(f"{cnt[s][cls]:>6}" for s in ("train", "val", "test")))

        extracted = safe_extract_zip_images(zf, img_out)
        print(f"解包图像 {extracted} 张 -> {img_out}")

    missing = [r["image_id"] for r in rows if not (img_out / f"{r['image_id']}.jpg").exists()]
    print(f"缺失图像: {len(missing)}")
    return {split: dict(counts) for split, counts in cnt.items()}


def main() -> None:
    """CLI 入口：从 Kaggle zip 准备 HAM10000 数据集与病灶级划分。"""
    ap = argparse.ArgumentParser(description="解包 HAM10000 zip 并按 lesion_id 分层划分")
    ap.add_argument("--zip", default="data/ham10000.zip")
    ap.add_argument("--out-dir", default="data/ham10000")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prepare_from_zip(
        zip_path=args.zip,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
