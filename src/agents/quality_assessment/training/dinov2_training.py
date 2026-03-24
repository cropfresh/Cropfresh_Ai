"""Training and export orchestration for commodity-conditioned DINO grading."""

from __future__ import annotations

import pathlib
import sys

from src.agents.quality_assessment.training.commodity_registry import COMMODITY_SPECS
from src.agents.quality_assessment.training.dinov2_artifacts import export_onnx, set_training_seed
from src.agents.quality_assessment.training.dinov2_data import build_dataloaders
from src.agents.quality_assessment.training.dinov2_loop import evaluate_model, train_epoch
from src.agents.quality_assessment.training.dinov2_metrics import (
    TrainingReport,
    write_training_report,
)
from src.agents.quality_assessment.training.dinov2_model import build_grade_model
from src.agents.quality_assessment.training.model_contracts import (
    load_validated_onnx_session,
    validate_dino_grade_session,
)
from src.shared.logger import setup_logger

logger = setup_logger()


def require_torch() -> None:
    """Fail early with a clear message when training deps are unavailable."""
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error("torch/transformers are required for DINO training.")
        sys.exit(1)


def train_manifest_model(
    manifest_path: pathlib.Path,
    output_path: pathlib.Path,
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
    patience: int = 5,
    label_smoothing: float = 0.05,
    unfreeze_backbone_layers: int = 1,
    balance_train: bool = True,
    report_path: pathlib.Path | None = None,
) -> None:
    """Train a commodity-conditioned DINO classifier from a manifest file."""
    import torch
    import torch.nn as nn

    set_training_seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = report_path or output_path.with_suffix(".metrics.json")
    checkpoint_path = output_path.with_name(f"{output_path.stem}-best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_dataloaders(
        manifest_path,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        balance_train=balance_train,
    )
    model = build_grade_model(
        num_commodities=max((spec.commodity_id for spec in COMMODITY_SPECS.values()), default=0)
        + 1,
        trainable_backbone_layers=unfreeze_backbone_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss(
        weight=bundle.class_weights.to(device),
        label_smoothing=label_smoothing,
    )
    best_epoch = 0
    best_macro_f1 = -1.0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, bundle, optimizer, criterion, device)
        metrics = evaluate_model(model, bundle, device)
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": round(train_loss, 6),
                "val_accuracy": metrics.accuracy,
                "val_macro_f1": metrics.macro_f1,
            }
        )
        logger.info(
            "DINO epoch {}/{} loss={:.4f} val_acc={:.4f} val_macro_f1={:.4f}",
            epoch,
            epochs,
            train_loss,
            metrics.accuracy,
            metrics.macro_f1,
        )
        if metrics.macro_f1 > best_macro_f1:
            best_epoch = epoch
            best_macro_f1 = metrics.macro_f1
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("Early stopping DINO training after {} stale epochs", stale_epochs)
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    validation = evaluate_model(model, bundle, device)
    export_onnx(model, output_path, device)
    load_validated_onnx_session(output_path, validate_dino_grade_session)
    write_training_report(
        report_path,
        TrainingReport(
            artifact_path=str(output_path),
            checkpoint_path=str(checkpoint_path),
            training_config={
                "epochs_requested": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "seed": seed,
                "num_workers": num_workers,
                "patience": patience,
                "label_smoothing": label_smoothing,
                "unfreeze_backbone_layers": unfreeze_backbone_layers,
                "balance_train": balance_train,
            },
            train_label_counts=bundle.train_label_counts,
            val_label_counts=bundle.val_label_counts,
            best_epoch=best_epoch,
            epochs_completed=len(history),
            history=history,
            validation=validation,
        ),
    )
    logger.info("Exported validated DINO classifier to {}", output_path)


def evaluate_checkpoint(
    manifest_path: pathlib.Path,
    checkpoint: pathlib.Path,
    batch_size: int = 16,
    num_workers: int = 0,
    report_path: pathlib.Path | None = None,
) -> None:
    """Evaluate a saved DINO checkpoint against the manifest val split."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_dataloaders(
        manifest_path, batch_size=batch_size, num_workers=num_workers, balance_train=False
    )
    model = build_grade_model(
        num_commodities=max((spec.commodity_id for spec in COMMODITY_SPECS.values()), default=0) + 1
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    validation = evaluate_model(model, bundle, device)
    logger.info(
        "DINO checkpoint accuracy={:.4f} macro_f1={:.4f}", validation.accuracy, validation.macro_f1
    )
    if report_path is not None:
        write_training_report(
            report_path,
            TrainingReport(
                artifact_path="",
                checkpoint_path=str(checkpoint),
                training_config={
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "mode": "eval_only",
                },
                train_label_counts=bundle.train_label_counts,
                val_label_counts=bundle.val_label_counts,
                best_epoch=0,
                epochs_completed=0,
                history=[],
                validation=validation,
            ),
        )
