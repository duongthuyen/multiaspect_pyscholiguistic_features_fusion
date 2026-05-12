"""
Train EndToEndFusionModel — RoBERTa backbone + fusion layers in one pass.

Usage
-----
    python -m scripts.models.e2e.train
    python -m scripts.models.e2e.train --epochs 5 --backbone-lr 2e-5 --fusion-lr 1e-4

The script:
  1. Loads text from processed CSVs and pre-extracted affective / handcrafted
     features from their parquet files.
  2. Fits a StandardScaler on training handcrafted features and applies it to
     val / test (mirrors the two-stage fusion pipeline).
  3. Trains with two AdamW parameter groups (backbone vs. fusion layers).
  4. Saves the best checkpoint (by val macro-F1) and evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from scripts import config
from scripts.evaluation.metrics import save_confusion_matrix_artifacts
from scripts.models.e2e.model import EndToEndFusionModel
from scripts.models.fusion.feature_loader import load_group_features
from scripts.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_handcrafted(split: str) -> np.ndarray:
    parts = []
    for group in ["lexical", "syntactic", "structural"]:
        _, mat = load_group_features(group, split=split)
        parts.append(mat)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _load_affective(split: str) -> np.ndarray:
    _, mat = load_group_features("affective", split=split)
    return mat.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class E2EDataset(Dataset):
    """Returns tokenised text + pre-extracted feature tensors + label."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        affective: np.ndarray,
        handcrafted: np.ndarray,
    ) -> None:
        self.texts = df[config.TEXT_COL].fillna("").astype(str).tolist()
        self.labels = df[config.LABEL_COL].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.affective = torch.from_numpy(affective)
        self.handcrafted = torch.from_numpy(handcrafted)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "affective": self.affective[idx],
            "handcrafted": self.handcrafted[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        affective = batch["affective"].to(device)
        handcrafted = batch["handcrafted"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, affective, handcrafted)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = float((np.array(all_preds) == np.array(all_labels)).mean())
    f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, f1


@torch.no_grad()
def run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, list, list]:
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        affective = batch["affective"].to(device)
        handcrafted = batch["handcrafted"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, affective, handcrafted)
        total_loss += criterion(logits, labels).item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = float((np.array(all_preds) == np.array(all_labels)).mean())
    f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    backbone_lr: float = config.E2E_BACKBONE_LR,
    fusion_lr: float = config.E2E_FUSION_LR,
    epochs: int = config.E2E_EPOCHS,
    batch_size: int = config.E2E_BATCH_SIZE,
    weight_decay: float = config.E2E_WEIGHT_DECAY,
    warmup_ratio: float = config.E2E_WARMUP_RATIO,
    grad_clip: float = config.E2E_GRAD_CLIP,
    seed: int = config.SEED,
    init_from_finetuned: bool = True,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = config.E2E_MODEL_DIR / "checkpoints"
    eval_dir = config.RESULTS_DIR / "e2e" / "evaluation"
    log_dir = config.RESULTS_DIR / "e2e" / "logs"
    for d in [ckpt_dir, eval_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=log_dir / "train.log")
    logger.info("=" * 60)
    logger.info("E2E fusion training  device=%s", device)
    logger.info(
        "backbone_lr=%g  fusion_lr=%g  epochs=%d  batch=%d  seed=%d",
        backbone_lr, fusion_lr, epochs, batch_size, seed,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    logger.info("Loading processed CSVs...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    logger.info("Loading pre-extracted features...")
    hc_tr = _load_handcrafted("train")
    hc_va = _load_handcrafted("val")
    hc_te = _load_handcrafted("test")

    scaler = StandardScaler()
    hc_tr = scaler.fit_transform(hc_tr).astype(np.float32)
    hc_va = scaler.transform(hc_va).astype(np.float32)
    hc_te = scaler.transform(hc_te).astype(np.float32)
    joblib.dump(scaler, ckpt_dir / "handcrafted_scaler.joblib")

    af_tr = _load_affective("train")
    af_va = _load_affective("val")
    af_te = _load_affective("test")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    backbone_path = (
        str(config.FINETUNED_ROBERTA_DIR)
        if (init_from_finetuned and config.FINETUNED_ROBERTA_DIR.exists())
        else config.MENTAL_ROBERTA_NAME
    )
    logger.info("Loading tokenizer from: %s", backbone_path)
    tokenizer = AutoTokenizer.from_pretrained(backbone_path)

    def make_loader(df, aff, hc, shuffle):
        return DataLoader(
            E2EDataset(df, tokenizer, config.MAX_LENGTH, aff, hc),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    train_loader = make_loader(train_df, af_tr, hc_tr, shuffle=True)
    val_loader = make_loader(val_df, af_va, hc_va, shuffle=False)
    test_loader = make_loader(test_df, af_te, hc_te, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Building EndToEndFusionModel (backbone=%s)...", backbone_path)
    model = EndToEndFusionModel(backbone_name_or_path=backbone_path).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{total_params:,}")

    # ── Optimizer — two LR groups ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": backbone_lr},
            {"params": model.fusion_parameters(), "lr": fusion_lr},
        ],
        weight_decay=weight_decay,
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Training loop ────────────────────────────────────────────────────────
    history: list[dict] = []
    best_val_f1 = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, grad_clip
        )
        va_loss, va_acc, va_f1, _, _ = run_eval_epoch(
            model, val_loader, criterion, device
        )
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d  train: loss=%.4f acc=%.4f f1=%.4f  "
            "val: loss=%.4f acc=%.4f f1=%.4f  (%.0fs)",
            epoch, epochs, tr_loss, tr_acc, tr_f1, va_loss, va_acc, va_f1, elapsed,
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
            "train_f1": round(tr_f1, 6),
            "val_loss": round(va_loss, 6), "val_acc": round(va_acc, 6),
            "val_f1": round(va_f1, 6),
        })

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_dir / "best.pt")
            logger.info("  -> Best checkpoint (val macro-F1=%.4f)", best_val_f1)

    # ── Test evaluation ──────────────────────────────────────────────────────
    logger.info("Evaluating best checkpoint (epoch %d) on test set...", best_epoch)
    model.load_state_dict(best_state)
    te_loss, te_acc, te_f1, te_preds, te_labels = run_eval_epoch(
        model, test_loader, criterion, device
    )
    logger.info("Test  loss=%.4f  acc=%.4f  macro-F1=%.4f", te_loss, te_acc, te_f1)

    class_names = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]
    report = classification_report(
        te_labels, te_preds, target_names=class_names, zero_division=0
    )
    logger.info("Classification report:\n%s", report)

    cm_artifacts = save_confusion_matrix_artifacts(
        np.array(te_labels), np.array(te_preds), class_names,
        eval_dir / "confusion_matrix",
    )

    # ── Save artifacts ───────────────────────────────────────────────────────
    summary = {
        "backbone": backbone_path,
        "backbone_lr": backbone_lr,
        "fusion_lr": fusion_lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 6),
        "test_loss": round(te_loss, 6),
        "test_acc": round(te_acc, 6),
        "test_macro_f1": round(te_f1, 6),
        "history": history,
        "confusion_matrix": cm_artifacts,
    }
    with open(eval_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Checkpoint -> %s", ckpt_dir / "best.pt")
    logger.info("Summary    -> %s", eval_dir / "summary.json")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-lr", type=float, default=config.E2E_BACKBONE_LR)
    parser.add_argument("--fusion-lr", type=float, default=config.E2E_FUSION_LR)
    parser.add_argument("--epochs", type=int, default=config.E2E_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.E2E_BATCH_SIZE)
    parser.add_argument(
        "--from-pretrained",
        action="store_true",
        help="Initialise from mental/mental-roberta-base instead of the finetuned backbone.",
    )
    args = parser.parse_args()

    train(
        backbone_lr=args.backbone_lr,
        fusion_lr=args.fusion_lr,
        epochs=args.epochs,
        batch_size=args.batch,
        init_from_finetuned=not args.from_pretrained,
    )
