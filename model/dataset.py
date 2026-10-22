"""数据集读取与图像预处理。"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import BASE_DIR
from constants import CATEGORY_ORDER

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = False):
    """训练使用数据增强，验证/测试只做缩放与中心裁剪。"""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    resize_size = int(round(image_size * 1.14 / 32) * 32)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class GarbageDataset(Dataset):
    """从 CSV（path,label,subclass）读取样本，label 为四分类编码。"""

    def __init__(
        self,
        csv_file: str | Path,
        image_size: int = 224,
        train: bool = False,
        root: str | Path = BASE_DIR,
    ):
        self.csv_file = Path(csv_file)
        self.root = Path(root)
        self.transform = build_transforms(image_size, train=train)

        with self.csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        if not rows:
            raise ValueError(f"数据列表为空：{self.csv_file}")
        if "path" not in rows[0] or "label" not in rows[0]:
            raise ValueError(f"CSV 缺少 path/label 列：{self.csv_file}")

        self.samples: list[tuple[Path, int]] = []
        for row in rows:
            label = row["label"].strip()
            if label not in CATEGORY_ORDER:
                raise ValueError(f"未知标签 {label!r}，应为 {CATEGORY_ORDER}")
            path = Path(row["path"])
            if not path.is_absolute():
                path = self.root / path
            self.samples.append((path, CATEGORY_ORDER.index(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(label, dtype=torch.long)

    def label_counts(self) -> list[int]:
        counts = [0] * len(CATEGORY_ORDER)
        for _, label in self.samples:
            counts[label] += 1
        return counts


def build_loaders(
    train_csv: str | Path,
    val_csv: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    root: str | Path = BASE_DIR,
):
    train_set = GarbageDataset(train_csv, image_size, train=True, root=root)
    val_set = GarbageDataset(val_csv, image_size, train=False, root=root)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_set) > batch_size,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_set, val_set
