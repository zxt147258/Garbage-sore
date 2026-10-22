"""模型训练脚本：支持轻量 ResNet、简单 CNN 与 ResNet18 迁移学习。

示例：
    python -m model.train --arch lightresnet --epochs 30 --batch-size 32 --lr 0.001
    python -m model.train --arch resnet18 --pretrained --epochs 15 --lr 0.0003
    python -m model.train --fold 1 --epochs 10 --out weights/fold_1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from config import BASE_DIR
from constants import CATEGORY_NAMES, CATEGORY_ORDER
from model.dataset import build_loaders
from model.net import build_model, count_parameters


@dataclass
class TrainConfig:
    arch: str = "lightresnet"
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 5e-4
    image_size: int = 224
    num_workers: int = 0
    patience: int = 6
    label_smoothing: float = 0.1
    seed: int = 42
    device: str = "auto"
    train_csv: str = "data/splits/train.csv"
    val_csv: str = "data/splits/val.csv"
    out_dir: str = "weights"
    run_name: str = "best"
    pretrained: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境不可用。")
    return device


def compute_metrics(confusion: torch.Tensor) -> tuple[float, float, list[dict]]:
    """由混淆矩阵计算总体准确率、宏平均 F1 与各类别指标。"""
    total = confusion.sum().item()
    accuracy = confusion.diag().sum().item() / total if total else 0.0

    per_class = []
    f1_scores = []
    for index, code in enumerate(CATEGORY_ORDER):
        tp = confusion[index, index].item()
        fp = confusion[:, index].sum().item() - tp
        fn = confusion[index, :].sum().item() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        per_class.append(
            {
                "code": code,
                "name": CATEGORY_NAMES[code],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int(confusion[index, :].sum().item()),
            }
        )
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return accuracy, macro_f1, per_class


def evaluate(model, loader, device, num_classes: int = 4, criterion=None) -> dict:
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if criterion is not None:
                total_loss += criterion(logits, labels).item() * labels.size(0)
            total_samples += labels.size(0)

            preds = logits.argmax(dim=1)
            for target, pred in zip(labels.cpu(), preds.cpu()):
                confusion[target.long(), pred.long()] += 1

    accuracy, macro_f1, per_class = compute_metrics(confusion)
    return {
        "loss": round(total_loss / total_samples, 4) if total_samples else 0.0,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else BASE_DIR / candidate


def run_training(config: TrainConfig, verbose: bool = True) -> dict:
    started = time.time()
    set_seed(config.seed)

    train_csv = _resolve(config.train_csv)
    val_csv = _resolve(config.val_csv)
    out_dir = _resolve(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"未找到数据列表：{train_csv} / {val_csv}，请先运行 model.prepare_data。"
        )

    device = resolve_device(config.device)
    train_loader, val_loader, train_set, val_set = build_loaders(
        train_csv,
        val_csv,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
    )

    model = build_model(config.arch, num_classes=len(CATEGORY_ORDER), pretrained=config.pretrained)
    model.to(device)

    counts = torch.tensor(train_set.label_counts(), dtype=torch.float32)
    class_weights = (counts.sum() / (len(CATEGORY_ORDER) * counts.clamp(min=1))).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=config.label_smoothing
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    if verbose:
        print("=" * 64)
        print(f"模型: {config.arch}    参数量: {count_parameters(model):,}")
        print(f"设备: {device}    训练集: {len(train_set)}    验证集: {len(val_set)}")
        print(f"超参数: lr={config.lr}  wd={config.weight_decay}  bs={config.batch_size}")
        print("=" * 64)

    history: list[dict] = []
    best_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_state: dict | None = None
    best_metrics: dict | None = None

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, criterion=criterion)
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "seconds": round(time.time() - epoch_start, 1),
        }
        history.append(record)
        if verbose:
            print(
                f"Epoch {epoch:02d}/{config.epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f}  {record['seconds']}s"
            )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                if verbose:
                    print(f"验证集连续 {config.patience} 轮未提升，提前停止。")
                break

    trained_at = datetime.now().astimezone().isoformat(timespec="seconds")
    checkpoint = {
        "arch": config.arch,
        "class_names": CATEGORY_ORDER,
        "image_size": config.image_size,
        "state_dict": best_state,
        "best_val_acc": round(best_accuracy, 4),
        "best_epoch": best_epoch,
        "trained_at": trained_at,
        "config": asdict(config),
        "metrics": best_metrics,
    }
    checkpoint_path = out_dir / f"{config.run_name}.pt"
    torch.save(checkpoint, checkpoint_path)

    meta = {
        "arch": config.arch,
        "class_names": CATEGORY_ORDER,
        "image_size": config.image_size,
        "best_val_acc": round(best_accuracy, 4),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "params": count_parameters(model),
        "trained_at": trained_at,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "class_counts": dict(zip(CATEGORY_ORDER, train_set.label_counts())),
        "config": asdict(config),
    }
    (out_dir / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed = time.time() - started
    if verbose:
        print("-" * 64)
        print(f"训练完成：best val_acc={best_accuracy:.4f} @ epoch {best_epoch}")
        print(f"权重已保存：{checkpoint_path}")
        print(f"总耗时：{elapsed / 60:.1f} 分钟")

    return {
        "checkpoint": str(checkpoint_path),
        "out_dir": str(out_dir),
        "best_val_acc": round(best_accuracy, 4),
        "best_epoch": best_epoch,
        "params": meta["params"],
        "elapsed_seconds": round(elapsed, 1),
        "arch": config.arch,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校园垃圾四分类模型训练")
    parser.add_argument("--arch", default="lightresnet", choices=["lightresnet", "simplecnn", "resnet18"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument("--out", default="weights")
    parser.add_argument("--run-name", default="best")
    parser.add_argument("--pretrained", action="store_true", help="仅 resnet18 生效")
    parser.add_argument("--fold", type=int, default=0, help="使用 K 折数据中的第 N 折")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.fold:
        fold_dir = BASE_DIR / "data" / "splits" / "cv" / f"fold_{args.fold}"
        args.train_csv = str(fold_dir / "train.csv")
        args.val_csv = str(fold_dir / "val.csv")
        if args.out == "weights":
            args.out = f"weights/fold_{args.fold}"

    config = TrainConfig(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        device=args.device,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        out_dir=args.out,
        run_name=args.run_name,
        pretrained=args.pretrained,
    )
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
