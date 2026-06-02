from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
try:
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
except Exception as e:
    raise ImportError("torchvision with EfficientNetV2-S is required.") from e

class SlovoEfficientNetV2S(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        drop: float = 0.3,
        weights_path: Path | str | None = None,
    ):
        super().__init__()
        if pretrained and weights_path is not None and Path(weights_path).exists():
            self.model = efficientnet_v2_s(weights=None)
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state, strict=True)
        elif pretrained:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            self.model = efficientnet_v2_s(weights=weights)
        else:
            self.model = efficientnet_v2_s(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(in_features, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
