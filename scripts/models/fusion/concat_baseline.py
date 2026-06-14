"""
Concatenation + MLP baseline for gated fusion comparison.

This model concatenates all three feature branches (semantic 768-dim,
affective 34-dim, handcrafted 26-dim) and passes them through a two-layer
MLP classifier. It is structurally equivalent to gated fusion with all gate
weights fixed to uniform (1/3 each) and no learned routing — serving as an
upper bound on "does gating help over naive combination?"

Architecture:
    concat([sem, aff, hc])  →  Linear(828, 256)  →  GELU  →  Dropout(0.3)
                             →  Linear(256, 6)

Usage:
    python -m scripts.models.fusion.concat_baseline

Outputs are written to:
    results/concat_mlp/evaluation/summary.json
    results/concat_mlp/evaluation/summary.txt
    results/concat_mlp/evaluation/classification_report.csv
    results/concat_mlp/evaluation/confusion_matrix/
    results/concat_mlp/training/training_history.csv
"""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts import config
from scripts.evaluation.metrics import save_confusion_matrix_artifacts
from scripts.models.fusion.feature_loader import load_feature_tensors
from scripts.models.fusion.train import _load_labels, _scale_hc, _accuracy
from scripts.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

_ROBERTA_BASELINE_F1 = 0.8873
OUTPUT_DIR = config.RESULTS_DIR / "concat_mlp"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConcatMLP(nn.Module):
    """
    Naive concatenation baseline.

    Concatenates all three feature branches (sem 768, aff 34, hc 26 = 828
    total) and classifies through a two-layer MLP.  No gating, no projection
    per branch.
    """

    def __init__(
        self,
        semantic_dim: int = config.SEMANTIC_DIM,
        affective_dim: int = config.AFFECTIVE_DIM,
        handcrafted_dim: int = config.HANDCRAFTED_DIM,
        hidden_dim: int = 256,
        num_labels: int = config.NUM_LABELS,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = semantic_dim + affective_dim + handcrafted_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        sem: torch.Tensor,
        aff: torch.Tensor,
        hc: torch.Tensor,
    ) -> torch.Tensor:
        return self.mlp(torch.cat([sem, aff, hc], dim=-1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _load_split(split: str):
    sem, aff, hc, _ = load_feature_tensors(input_config="fused", split=split)
    labels = _load_labels(split)
    return sem, aff, hc, labels


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for sem, aff, hc, labels in loader:
            sem = sem.to(device)
            aff = aff.to(device)
            hc = hc.to(device)
            labels = labels.to(device)

            logits = model(sem, aff, hc)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            total_acc += _accuracy(logits.detach(), labels)
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sem, aff, hc, labels in loader:
            logits = model(sem.to(device), aff.to(device), hc.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def train() -> dict:
    seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = OUTPUT_DIR / "training" / "checkpoints"
    log_dir = OUTPUT_DIR / "training" / "logs"
    eval_dir = OUTPUT_DIR / "evaluation"
    for p in [ckpt_dir, log_dir, eval_dir]:
        p.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=log_dir / "train.log")
    logger.info("=" * 60)
    logger.info("Concat+MLP baseline  device=%s", device)
    logger.info("=" * 60)

    logger.info("Loading features...")
    t0 = time.time()
    sem_tr, aff_tr, hc_tr, lbl_tr = _load_split("train")
    sem_va, aff_va, hc_va, lbl_va = _load_split("val")
    sem_te, aff_te, hc_te, lbl_te = _load_split("test")
    logger.info(
        "Loaded train=%d val=%d test=%d (%.1fs)",
        len(lbl_tr), len(lbl_va), len(lbl_te), time.time() - t0,
    )

    scaler = StandardScaler()
    hc_tr = _scale_hc(scaler, hc_tr, fit=True)
    hc_va = _scale_hc(scaler, hc_va)
    hc_te = _scale_hc(scaler, hc_te)

    batch_size = config.FUSION_BATCH_SIZE
    train_loader = DataLoader(
        TensorDataset(sem_tr, aff_tr, hc_tr, lbl_tr),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(sem_va, aff_va, hc_va, lbl_va),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(sem_te, aff_te, hc_te, lbl_te),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model = ConcatMLP().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("ConcatMLP params=%s", f"{n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.FUSION_LR)

    class_names = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]
    history: list[dict] = []

    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    patience = config.FUSION_EARLY_STOPPING_PATIENCE

    for epoch in range(1, config.FUSION_EPOCHS + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, None, device)

        logger.info(
            "Epoch %d/%d  train: loss=%.4f acc=%.4f  val: loss=%.4f acc=%.4f",
            epoch, config.FUSION_EPOCHS, tr_loss, tr_acc, va_loss, va_acc,
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "train_acc": round(tr_acc, 6),
            "val_loss": round(va_loss, 6),
            "val_acc": round(va_acc, 6),
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    te_loss, te_acc = _run_epoch(model, test_loader, criterion, None, device)
    preds, true_labels = _predict(model, test_loader, device)

    macro_f1 = float(f1_score(true_labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(true_labels, preds, average="weighted", zero_division=0))
    per_class_f1 = f1_score(true_labels, preds, average=None, zero_division=0)

    report_str = classification_report(true_labels, preds, target_names=class_names, zero_division=0)
    logger.info("Test loss=%.4f acc=%.4f macro_f1=%.6f", te_loss, te_acc, macro_f1)
    logger.info("Classification report:\n%s", report_str)

    cm = save_confusion_matrix_artifacts(
        true_labels, preds, class_names, eval_dir / "confusion_matrix"
    )

    # Save classification report CSV
    clf_rpt = classification_report(
        true_labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    clf_path = eval_dir / "classification_report.csv"
    with open(clf_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class", "precision", "recall", "f1_score", "support"]
        )
        writer.writeheader()
        for cls in class_names:
            r = clf_rpt[cls]
            writer.writerow({
                "class": cls,
                "precision": f"{r['precision']:.6f}",
                "recall": f"{r['recall']:.6f}",
                "f1_score": f"{r['f1-score']:.6f}",
                "support": int(r["support"]),
            })

    # Save history CSV
    hist_path = OUTPUT_DIR / "training" / "training_history.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)

    results = {
        "model_name": "concat_mlp",
        "model_class": "ConcatMLP",
        "input_dim": config.SEMANTIC_DIM + config.AFFECTIVE_DIM + config.HANDCRAFTED_DIM,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "lr": config.FUSION_LR,
        "batch_size": batch_size,
        "seed": seed,
        "label_smoothing": 0.1,
        "best_val_acc": round(best_val_acc, 6),
        "test_acc": round(te_acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class_f1": {
            cls: round(float(per_class_f1[i]), 6) for i, cls in enumerate(class_names)
        },
        "history": history,
        "confusion_matrix": cm,
    }

    json_path = eval_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    txt_lines = [
        "Concat+MLP Baseline",
        "=" * 50,
        f"Input dim             : {results['input_dim']} (sem={config.SEMANTIC_DIM}, aff={config.AFFECTIVE_DIM}, hc={config.HANDCRAFTED_DIM})",
        f"Epochs trained        : {results['epochs_trained']}  (best: {results['best_epoch']})",
        f"Best val acc          : {results['best_val_acc']:.4f}",
        f"Test accuracy         : {results['test_acc']:.4f}",
        f"Test macro F1         : {results['macro_f1']:.6f}",
        f"Test weighted F1      : {results['weighted_f1']:.4f}",
        "",
        "Per-class F1 (test):",
    ]
    for cls, f1v in results["per_class_f1"].items():
        delta = f1v - (clf_rpt[cls]["f1-score"] if cls in clf_rpt else 0)
        txt_lines.append(f"  {cls:<12} {f1v:.4f}")
    txt_lines += [
        "",
        f"vs MentalRoBERTa baseline macro F1 = {_ROBERTA_BASELINE_F1:.4f}",
        f"  delta = {macro_f1 - _ROBERTA_BASELINE_F1:+.4f}",
    ]

    (eval_dir / "summary.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    print("\n=== CONCAT+MLP BASELINE ===")
    print(f"Test macro F1 : {macro_f1:.6f}")
    print(f"Test accuracy : {te_acc:.4f}")
    print(f"vs RoBERTa    : {macro_f1 - _ROBERTA_BASELINE_F1:+.6f}")
    print("\nPer-class F1:")
    for cls, f1v in results["per_class_f1"].items():
        print(f"  {cls:<12} {f1v:.4f}")

    return results


if __name__ == "__main__":
    train()
