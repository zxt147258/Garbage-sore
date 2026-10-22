"""数据清洗、标签归并与数据集划分。

用法示例：
    python -m model.prepare_data --raw data/raw --out data/splits
    python -m model.prepare_data --raw data/raw --cv-folds 5

输出：
    data/splits/train.csv  val.csv  test.csv
    data/splits/summary.json
    data/splits/cv/fold_1/...（可选）
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from config import BASE_DIR, DATA_DIR
from constants import CATEGORY_ORDER
from model.labeling import resolve_category

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MAPPING = DATA_DIR / "label_mapping.json"


def load_mapping(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    return payload.get("label_map", payload) if isinstance(payload, dict) else {}


def load_garbage_label(raw_dir: Path) -> dict[str, str]:
    """读取华为云垃圾数据集自带的 garbage_label.txt（数字目录 -> 中文名）。"""
    label_file = next(raw_dir.rglob("garbage_label.txt"), None)
    if label_file is None:
        return {}

    mapping: dict[str, str] = {}
    for encoding in ("utf-8-sig", "gbk"):
        try:
            lines = label_file.read_text(encoding=encoding).splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"[警告] 无法解码 {label_file}，已忽略。")
        return {}

    for line in lines:
        match = re.match(r"^\s*(\d+)\s*[:：,，\t ]\s*(.+?)\s*$", line)
        if match:
            mapping[match.group(1)] = match.group(2)
    return mapping


def iter_images(raw_dir: Path):
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(part.startswith(".") or part == "__MACOSX" for part in path.parts):
            continue
        yield path


def inspect_image(path: Path, min_size: int) -> tuple[bool, str]:
    try:
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            image.verify()
    except Exception:
        return False, "invalid"

    if width < min_size or height < min_size:
        return False, "too_small"
    return True, hashlib.md5(raw).hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def split_group(
    items: list[dict], rng: random.Random, val_ratio: float, test_ratio: float
) -> tuple[list[dict], list[dict], list[dict]]:
    """按子类别分层划分，样本过少时全部留给训练集。"""
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)

    if n < 5:
        return shuffled, [], []

    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    while n - n_test - n_val < 1:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write("path,label,subclass\n")
        for row in rows:
            handle.write(f'{row["path"]},{row["label"]},{row["subclass"]}\n')


def count_by_category(rows: list[dict]) -> dict[str, int]:
    counter = Counter(row["label"] for row in rows)
    return {code: counter.get(code, 0) for code in CATEGORY_ORDER}


def prepare(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    if not raw_dir.is_absolute():
        raw_dir = BASE_DIR / raw_dir
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir

    if not raw_dir.exists():
        print(f"[错误] 数据目录不存在：{raw_dir}")
        print("请将公开数据集解压到 data/raw/<子类别>/*.jpg 后重试。")
        return 1

    mapping_path = Path(args.mapping) if args.mapping else DEFAULT_MAPPING
    if not mapping_path.is_absolute():
        mapping_path = BASE_DIR / mapping_path
    explicit_map = load_mapping(mapping_path)
    id_to_text = load_garbage_label(raw_dir)

    print(f"扫描目录：{raw_dir}")
    if explicit_map:
        print(f"已加载显式映射：{mapping_path}（{len(explicit_map)} 条）")
    if id_to_text:
        print(f"已加载 garbage_label.txt（{len(id_to_text)} 条数字目录映射）")

    rows: list[dict] = []
    skipped = Counter()
    unlabeled = Counter()
    seen_hashes: set[str] = set()

    scanned = 0
    for path in iter_images(raw_dir):
        scanned += 1
        if scanned % 500 == 0:
            print(f"  已扫描 {scanned} 张...")

        folder = path.parent.name
        label_text = id_to_text.get(folder, folder) if folder.isdigit() else folder
        code = resolve_category(label_text, explicit_map=explicit_map)
        if code is None:
            skipped["unlabeled"] += 1
            unlabeled[folder] += 1
            continue

        valid, digest = inspect_image(path, args.min_size)
        if not valid:
            skipped[digest] += 1
            continue
        if not args.keep_duplicates:
            if digest in seen_hashes:
                skipped["duplicate"] += 1
                continue
            seen_hashes.add(digest)

        rows.append(
            {
                "path": relative_path(path),
                "label": code,
                "subclass": label_text,
            }
        )

    if not rows:
        print("[错误] 没有可用样本。请检查目录命名是否能映射到四分类。")
        print("可用中文/英文子类别名，或通过 data/label_mapping.json 显式配置。")
        return 1

    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_group[(row["label"], row["subclass"])].append(row)

    rng = random.Random(args.seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    for group in sorted(by_group):
        train, val, test = split_group(by_group[group], rng, args.val_ratio, args.test_ratio)
        train_rows.extend(train)
        val_rows.extend(val)
        test_rows.extend(test)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)
    write_csv(out_dir / "test.csv", test_rows)

    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "raw_dir": str(raw_dir),
        "total_used": len(rows),
        "total_scanned": scanned,
        "skipped": dict(skipped),
        "unlabeled_folders": dict(unlabeled.most_common(20)),
        "counts": {
            "all": count_by_category(rows),
            "train": count_by_category(train_rows),
            "val": count_by_category(val_rows),
            "test": count_by_category(test_rows),
        },
        "subclasses": len(by_group),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }

    if args.cv_folds and args.cv_folds >= 2:
        summary["cv_folds"] = args.cv_folds
        fold_of_row: dict[int, int] = {}
        for group in sorted(by_group):
            shuffled = by_group[group][:]
            rng.shuffle(shuffled)
            for index, row in enumerate(shuffled):
                fold_of_row[id(row)] = index % args.cv_folds

        num_written = 0
        for fold in range(args.cv_folds):
            fold_val = [row for row in rows if fold_of_row[id(row)] == fold]
            fold_train = [row for row in rows if fold_of_row[id(row)] != fold]
            if not fold_val or not fold_train:
                continue
            fold_dir = out_dir / "cv" / f"fold_{fold + 1}"
            rng.shuffle(fold_train)
            rng.shuffle(fold_val)
            write_csv(fold_dir / "train.csv", fold_train)
            write_csv(fold_dir / "val.csv", fold_val)
            num_written += 1
        print(f"已生成 {num_written} 折交叉验证数据：{out_dir / 'cv'}")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n数据准备完成")
    print(f"  样本总数：{len(rows)}（扫描 {scanned}，跳过 {sum(skipped.values())}）")
    for key in ("all", "train", "val", "test"):
        counts = summary["counts"][key]
        detail = "，".join(f"{code}={counts[code]}" for code in CATEGORY_ORDER)
        print(f"  {key:<5}: {detail}")
    if unlabeled:
        top = "，".join(f"{name}({count})" for name, count in unlabeled.most_common(5))
        print(f"  未识别目录（前 5）：{top}")
    print(f"  输出目录：{out_dir}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="垃圾图片数据清洗与划分")
    parser.add_argument("--raw", default="data/raw", help="原始数据目录")
    parser.add_argument("--out", default="data/splits", help="划分结果目录")
    parser.add_argument("--mapping", default=None, help="显式标签映射 JSON")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=0, help="生成 K 折交叉验证数据，0 为关闭")
    parser.add_argument("--min-size", type=int, default=64, help="最小边像素")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-duplicates", action="store_true", help="保留重复图片")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(prepare(parse_args()))
