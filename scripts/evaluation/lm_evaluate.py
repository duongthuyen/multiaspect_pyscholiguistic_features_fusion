"""Evaluation helpers for language-model classifiers such as MentalRoBERTa."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from scripts import config
from scripts.evaluation.metrics import save_confusion_matrix_artifacts

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, list[int], list[int]]:
    """Run one evaluation pass and return aggregate metrics plus predictions."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = float((np.array(all_preds) == np.array(all_labels)).mean())
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1, all_preds, all_labels


def evaluate_mental_roberta_checkpoint(
    checkpoint_dir: Path,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    best_val_f1: float | None = None,
) -> tuple[dict, nn.Module]:
    """Load the best MentalRoBERTa checkpoint, evaluate it, and save artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading best checkpoint for test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir)
    ).to(device)
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate_epoch(
        best_model,
        test_loader,
        device,
    )

    if best_val_f1 is None:
        logger.info(
            "Test  loss=%.4f  acc=%.4f  macro-F1=%.4f",
            test_loss,
            test_acc,
            test_f1,
        )
    else:
        logger.info(
            "Test  loss=%.4f  acc=%.4f  macro-F1=%.4f  best-val-macro-F1=%.4f",
            test_loss,
            test_acc,
            test_f1,
            best_val_f1,
        )

    class_names = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]
    report_str = classification_report(
        test_labels,
        test_preds,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        test_labels,
        test_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    logger.info("Classification report:\n%s", report_str)

    report_txt_path = output_dir / "classification_report.txt"
    report_txt_path.write_text(report_str, encoding="utf-8")

    report_json_path = output_dir / "classification_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    cm_artifacts = save_confusion_matrix_artifacts(
        np.array(test_labels),
        np.array(test_preds),
        class_names,
        output_dir / "confusion_matrix",
    )

    return (
        {
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
            "test_macro_f1": round(test_f1, 6),
            "classification_report": {
                "text_path": str(report_txt_path),
                "json_path": str(report_json_path),
            },
            "confusion_matrix": cm_artifacts,
        },
        best_model,
    )
