"""ResNet similarity model definition for Digital Twin verification."""

from __future__ import annotations


def build_similarity_model(embed_dim: int = 128):
    """Build a ResNet50 encoder with a normalized projection head."""
    import torch.nn as nn
    from torchvision import models

    class L2Norm(nn.Module):
        def forward(self, x):
            import torch.nn.functional as F

            return F.normalize(x, p=2, dim=1)

    class ProduceSimilarityNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backbone = models.resnet50(weights="IMAGENET1K_V2")
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.projection = nn.Sequential(nn.Flatten(), nn.Linear(2048, embed_dim), L2Norm())

        def forward(self, x):
            return self.projection(self.features(x))

    return ProduceSimilarityNet()
