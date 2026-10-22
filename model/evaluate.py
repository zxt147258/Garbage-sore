"""在测试集上评估模型，输出分类指标与混淆矩阵。

用法：
    python -m model.evaluate --checkpoint weights/best.pt --test-csv data/splits/test.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import BASE_DIR
from constants import CATEGORY_NAMES, CATEGORY_ORDER
from model.dataset import GarbageDataset
from model.net import build_model
from model.train import evaluate


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else BASE_DIR / candidate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="模型测试集评估")
    parser.add_argument("--checkpoint", default="weights/best.pt")
    parser.add_argument("--test-csv", default="data/splits/test.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    checkpoint_path = resolve(args.checkpoint)
    test_csv = resolve(args.test_csv)
    if not checkpoint_path.exists():
        print(f"[错误] 找不到权重文件：{checkpoint_path}")
        return 1
    if not test_csv.exists():
        print(f"[错误] 找不到测试集列表：{test_csv}")
        return 1

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    class_names = checkpoint.get("class_names", CATEGORY_ORDER)
    image_size = int(checkpoint.get("image_size", 224))
    arch = checkpoint.get("arch", "lightresnet")

    model = build_model(arch, num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    test_set = GarbageDataset(test_csv, image_size=image_size, train=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metrics = evaluate(model, test_loader, device, num_classes=len(class_names), criterion=nn.CrossEntropyLoss())

    print(f"模型：{arch}  图像尺寸：{image_size}  测试样本：{len(test_set)}")
    print(f"准确率：{metrics['accuracy']:.4f}  宏平均 F1：{metrics['macro_f1']:.4f}")
    print("-" * 62)
    print(f"{'类别':<10}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
    for item in metrics["per_class"]:
        print(
            f"{item['name']:<10}{item['precision']:>12.4f}{item['recall']:>10.4f}"
            f"{item['f1']:>10.4f}{item['support']:>10d}"
        )

    print("\n混淆矩阵（行=真实，列=预测）")
    header = " " * 10 + "".join(f"{CATEGORY_NAMES[code]:>10}" for code in CATEGORY_ORDER)
    print(header)
    for index, code in enumerate(CATEGORY_ORDER):
        row = "".join(f"{value:>10d}" for value in metrics["confusion"][index])
        print(f"{CATEGORY_NAMES[code]:<10}{row}")

    report = {
        "checkpoint": str(checkpoint_path),
        "test_csv": str(test_csv),
        "arch": arch,
        "image_size": image_size,
        "samples": len(test_set),
        **metrics,
    }
    out_path = Path(args.out) if args.out else checkpoint_path.parent / "eval_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n评估报告已保存：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
