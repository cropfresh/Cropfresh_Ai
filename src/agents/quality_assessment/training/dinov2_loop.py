"""Train/eval loop helpers for DINO quality-grading models."""

from __future__ import annotations

from src.agents.quality_assessment.training.dinov2_data import DinoDataBundle
from src.agents.quality_assessment.training.dinov2_metrics import evaluate_predictions


def train_epoch(model, bundle: DinoDataBundle, optimizer, criterion, device) -> float:
    """Run one training epoch and return the mean loss."""
    import torch

    model.train()
    total_loss = 0.0
    for images, commodity_ids, labels in bundle.train_loader:
        images, commodity_ids, labels = (
            images.to(device),
            commodity_ids.to(device),
            labels.to(device),
        )
        optimizer.zero_grad()
        loss = criterion(model(images, commodity_ids), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(bundle.train_loader), 1)


def evaluate_model(model, bundle: DinoDataBundle, device):
    """Run the validation loop and compute exact aggregate metrics."""
    import torch

    labels: list[int] = []
    predictions: list[int] = []
    commodity_ids: list[int] = []
    model.eval()
    with torch.no_grad():
        for images, batch_commodity_ids, batch_labels in bundle.val_loader:
            logits = model(images.to(device), batch_commodity_ids.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())
            commodity_ids.extend(batch_commodity_ids.tolist())
    return evaluate_predictions(labels, predictions, commodity_ids)
