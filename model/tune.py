"""小规模网格搜索调参，结果写入 CSV，便于报告展示。

示例：
    python -m model.tune --epochs 8 --archs lightresnet simplecnn \
        --lrs 0.001 0.0005 --weight-decays 0.0001 0.0005
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch

from config import BASE_DIR
from model.train import TrainConfig, run_training


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="超参数网格搜索")
    parser.add_argument("--archs", nargs="+", default=["lightresnet", "simplecnn"])
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 5e-4])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1e-4, 5e-4])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--out", default="weights/tune")
    args = parser.parse_args(argv)

    splits_dir = BASE_DIR / args.splits_dir
    out_root = BASE_DIR / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    run_index = 0
    total = len(args.archs) * len(args.lrs) * len(args.weight_decays)

    for arch in args.archs:
        for lr in args.lrs:
            for weight_decay in args.weight_decays:
                run_index += 1
                tag = f"{arch}_lr{lr:g}_wd{weight_decay:g}"
                print(f"\n[{run_index}/{total}] {tag}")
                config = TrainConfig(
                    arch=arch,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    image_size=args.image_size,
                    patience=args.patience,
                    seed=args.seed,
                    train_csv=str(splits_dir / "train.csv"),
                    val_csv=str(splits_dir / "val.csv"),
                    out_dir=str(out_root / tag),
                    run_name="best",
                )
                summary = run_training(config, verbose=True)
                summary["tag"] = tag
                results.append(summary)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    results.sort(key=lambda item: item["best_val_acc"], reverse=True)
    csv_path = out_root / "tune_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tag",
                "arch",
                "lr",
                "weight_decay",
                "batch_size",
                "best_val_acc",
                "best_epoch",
                "params",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow({key: item[key] for key in writer.fieldnames})

    (out_root / "tune_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n调参结果（按验证集准确率排序）")
    for item in results:
        print(f"  {item['best_val_acc']:.4f}  {item['tag']}")
    print(f"\n结果已保存：{csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
