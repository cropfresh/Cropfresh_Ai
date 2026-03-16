"""Commodity-conditioned DINOv2 grade-classifier model definition."""

from __future__ import annotations


def build_grade_model(
    num_classes: int = 4,
    num_commodities: int = 16,
    commodity_dim: int = 16,
    trainable_backbone_layers: int = 0,
):
    """Build a frozen DINOv2 backbone with a commodity-aware grade head."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class CommodityConditionedGradeClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
            _freeze_backbone(self.backbone)
            _unfreeze_backbone_tail(self.backbone, trainable_backbone_layers)
            self.commodity_embedding = nn.Embedding(num_commodities, commodity_dim)
            self.head = nn.Sequential(
                nn.Linear(384 + commodity_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, pixel_values: torch.Tensor, commodity_id: torch.Tensor):
            features = self.backbone(pixel_values=pixel_values).pooler_output
            commodity_features = self.commodity_embedding(commodity_id.long())
            if commodity_features.ndim == 3:
                commodity_features = commodity_features.squeeze(1)
            return self.head(torch.cat([features, commodity_features], dim=1))

    return CommodityConditionedGradeClassifier()


def _freeze_backbone(backbone) -> None:
    for param in backbone.parameters():
        param.requires_grad = False


def _unfreeze_backbone_tail(backbone, trainable_backbone_layers: int) -> None:
    if trainable_backbone_layers <= 0:
        return
    encoder = getattr(getattr(backbone, "encoder", None), "layer", None)
    if encoder is None:
        raise ValueError("DINO backbone does not expose encoder.layer for partial fine-tuning")
    for block in list(encoder)[-trainable_backbone_layers:]:
        for param in block.parameters():
            param.requires_grad = True
    layernorm = getattr(backbone, "layernorm", None)
    if layernorm is not None:
        for param in layernorm.parameters():
            param.requires_grad = True
