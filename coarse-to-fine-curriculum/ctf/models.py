from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50


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
    ) -> None:
        super().__init__()
        channels, height, width = input_shape
        layers: list[nn.Module] = [
            nn.Conv2d(channels, 32, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Conv2d(32, 64, kernel_size=3),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [nn.Conv2d(64, 64, kernel_size=3), nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            feature_dim = int(self.encoder(dummy).reshape(1, -1).shape[1])

        self.head = nn.Linear(feature_dim, num_classes)
        self.spec = ModelSpec(name="cnn", feature_dim=feature_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    @property
    def classifier_weight(self) -> torch.Tensor:
        return self.head.weight


class ResNetClassifier(nn.Module):
    def __init__(self, arch: str, num_classes: int) -> None:
        super().__init__()
        if arch == "resnet18":
            base = resnet18(weights=None)
        elif arch == "resnet50":
            base = resnet50(weights=None)
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
) -> nn.Module:
    if model_name == "cnn":
        return SmallCNN(input_shape=input_shape, num_classes=num_classes, dropout=dropout)
    if model_name in {"resnet18", "resnet50"}:
        return ResNetClassifier(arch=model_name, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")
