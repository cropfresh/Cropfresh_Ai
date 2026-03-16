"""Commodity-conditioned DINOv2 grade-classifier model definition."""

from __future__ import annotations


def build_grade_model(
    num_classes: int = 4,
    num_commodities: int = 16,
    commodity_dim: int = 16,
):
    """Build a frozen DINOv2 backbone with a commodity-aware grade head."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class CommodityConditionedGradeClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
            for param in self.backbone.parameters():
                param.requires_grad = False
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
