from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_dim: int


class SmallCNN(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        dropout: float = 0.0,
        width_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        channels, height, width = input_shape
        c1 = max(1, round(32 * width_multiplier))
        c2 = max(1, round(64 * width_multiplier))
        c3 = max(1, round(64 * width_multiplier))
        layers: list[nn.Module] = [
            nn.Conv2d(channels, c1, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Conv2d(c1, c2, kernel_size=3),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [nn.Conv2d(c2, c3, kernel_size=3), nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            feature_dim = int(self.encoder(dummy).reshape(1, -1).shape[1])

        self.head = nn.Linear(feature_dim, num_classes)
        self.spec = ModelSpec(name=f"cnn_w{width_multiplier:g}", feature_dim=feature_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    @property
    def classifier_weight(self) -> torch.Tensor:
        return self.head.weight


class CifarBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, depth: int, num_classes: int, width_multiplier: float = 1.0) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0 or depth < 8:
            raise ValueError("CIFAR ResNet depth must be 6n+2 and >= 8, e.g. 8, 14, 20, 32, 44, 56")
        blocks_per_stage = (depth - 2) // 6
        base = max(1, round(16 * width_multiplier))
        widths = [base, base * 2, base * 4]
        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(widths[0], blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(widths[1], blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(widths[2], blocks_per_stage, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(widths[2], num_classes)
        self.spec = ModelSpec(name=f"cifar_resnet{depth}_w{width_multiplier:g}", feature_dim=widths[2])

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(CifarBasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        return out.reshape(out.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    @property
    def classifier_weight(self) -> torch.Tensor:
        return self.head.weight


class ResNetClassifier(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained_backbone: bool = False) -> None:
        super().__init__()
        if arch == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
            base = resnet18(weights=weights)
        elif arch == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
            base = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet architecture: {arch}")

        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        feature_dim = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.head = nn.Linear(feature_dim, num_classes)
        self.spec = ModelSpec(name=arch, feature_dim=feature_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    @property
    def classifier_weight(self) -> torch.Tensor:
        return self.head.weight


def build_model(
    model_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout: float = 0.0,
    cnn_width_multiplier: float = 1.0,
    cifar_resnet_width_multiplier: float = 1.0,
    pretrained_backbone: bool = False,
) -> nn.Module:
    if model_name == "cnn":
        return SmallCNN(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout=dropout,
            width_multiplier=cnn_width_multiplier,
        )
    if model_name.startswith("cifar_resnet"):
        depth = int(model_name.removeprefix("cifar_resnet"))
        return CifarResNet(
            depth=depth,
            num_classes=num_classes,
            width_multiplier=cifar_resnet_width_multiplier,
        )
    if model_name in {"resnet18", "resnet50"}:
        return ResNetClassifier(
            arch=model_name,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
        )
    raise ValueError(f"Unsupported model: {model_name}")
