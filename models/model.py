"""GTSRB classifier: pretrained ResNet18 backbone, fine-tuned.

Early stages (conv1, bn1, layer1, layer2) are frozen and kept in eval mode
(so their BatchNorm running stats don't drift), while layer3, layer4 and the
classifier head are fine-tuned on GTSRB. The 512-d penultimate embedding
(output of the frozen/fine-tuned backbone, before the classifier head) is
exposed via `return_features=True` for use as the feature space in the
Mahalanobis OOD detector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

# ImageNet normalization stats: required for the pretrained backbone, and
# reused for every dataset (GTSRB, CIFAR-10) so OOD comparisons stay on the
# same input distribution the backbone was trained on.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_FREEZE = ("conv1", "bn1", "layer1", "layer2")


class GTSRBResNet18(nn.Module):
    def __init__(self, num_classes: int = 43, pretrained: bool = True,
                 freeze: tuple[str, ...] = DEFAULT_FREEZE):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._frozen_names = list(freeze)
        self._frozen_modules = [getattr(self.backbone, name) for name in self._frozen_names]
        for module in self._frozen_modules:
            for p in module.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # Keep frozen stages (incl. their BatchNorm running stats) in eval mode.
            for module in self._frozen_modules:
                module.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.backbone(x)  # (B, feature_dim), avgpool+flatten already applied
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def build_model(num_classes: int = 43, pretrained: bool = True,
                 freeze: tuple[str, ...] = DEFAULT_FREEZE) -> GTSRBResNet18:
    return GTSRBResNet18(num_classes=num_classes, pretrained=pretrained, freeze=freeze)


def save_checkpoint(path, model: GTSRBResNet18, extra: dict | None = None):
    payload = {
        "model_state": model.state_dict(),
        "num_classes": model.classifier.out_features,
        "feature_dim": model.feature_dim,
        "freeze": model._frozen_names,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[GTSRBResNet18, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(num_classes=payload["num_classes"], pretrained=False,
                         freeze=tuple(payload.get("freeze", DEFAULT_FREEZE)))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload
