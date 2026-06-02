from __future__ import annotations
import torch
import torch.nn as nn
try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except Exception as e:
    raise ImportError(
        "Не найден torchvision с EfficientNet. Установи torchvision: pip install torchvision"
    ) from e

class SlovoEfficientNetB0(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False, drop: float = 0.2):
        super().__init__()
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.model = efficientnet_b0(weights=weights)
        else:
            self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(in_features, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
