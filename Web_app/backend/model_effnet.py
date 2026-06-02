from __future__ import annotations
import torch
import torch.nn as nn
try:
    from torchvision.models import efficientnet_b0, efficientnet_v2_s
except Exception as e:
    raise ImportError(
        "torchvision with EfficientNet is required. Install torchvision."
    ) from e

class SlovoEfficientNetB0(nn.Module):
    def __init__(self, num_classes: int, drop: float = 0.3):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(in_features, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class SlovoEfficientNetV2S(nn.Module):
    def __init__(self, num_classes: int, drop: float = 0.3):
        super().__init__()
        self.model = efficientnet_v2_s(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(in_features, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
