"""模型定义：轻量 ResNet（默认）与简单 CNN 基线。"""

from __future__ import annotations

from torch import nn


class BasicBlock(nn.Module):
    """两层 3x3 卷积的残差块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class LightResNet(nn.Module):
    """面向四分类的轻量残差网络，参数量约 2.7M，可在 CPU 上较快推理。"""

    def __init__(
        self,
        num_classes: int = 4,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        blocks: tuple[int, ...] = (1, 1, 2, 2),
        dropout: float = 0.2,
    ):
        super().__init__()
        if len(widths) != 4 or len(blocks) != 4:
            raise ValueError("widths 与 blocks 均需包含 4 个阶段")

        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )

        stages = []
        in_channels = widths[0]
        for i, (width, num_blocks) in enumerate(zip(widths, blocks)):
            stride = 1 if i == 0 else 2
            for j in range(num_blocks):
                stages.append(BasicBlock(in_channels, width, stride if j == 0 else 1))
                in_channels = width
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(widths[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


class SimpleCNN(nn.Module):
    """课程基线模型：4 组卷积 + 全局平均池化。"""

    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        channels = (32, 64, 128, 256)
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


def build_model(arch: str, num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    """按名称构建模型。

    arch: lightresnet | simplecnn | resnet18
    pretrained 仅对 resnet18 生效（ImageNet 迁移学习）。
    """
    arch = arch.lower()
    if arch == "lightresnet":
        return LightResNet(num_classes=num_classes)
    if arch == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    if arch == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        try:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)
        except Exception as exc:  # 预训练权重下载失败时给出明确提示
            raise RuntimeError(
                "ResNet18 预训练权重下载失败，请检查网络，或改用 --arch lightresnet。"
            ) from exc
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"未知模型结构: {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
