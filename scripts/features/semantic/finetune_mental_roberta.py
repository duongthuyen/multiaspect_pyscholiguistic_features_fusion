"""
Fine-tune MentalRoBERTa on the 6-class mental health classification task,
then extract CLS embeddings from the fine-tuned backbone.

Pipeline
--------
1. Load processed CSVs (train / val / test) with columns ``text`` and ``class_id``.
2. Fine-tune ``AutoModelForSequenceClassification`` using AdamW + linear LR
   warm-up schedule (best checkpoint selected by validation macro-F1).
3. Save the best checkpoint (full model with classification head) to
   ``ROBERTA_MODEL_DIR / checkpoints / best_model``.
4. Save the bare backbone (no classification head) to ``FINETUNED_ROBERTA_DIR``
   so ``MentalRobertaExtractor(model_dir=FINETUNED_ROBERTA_DIR)`` can load it.
5. Extract CLS embeddings for every split and write them to the semantic
   feature parquet files, replacing any previously extracted embeddings.

Usage
-----
    python -m scripts.features.semantic.finetune_mental_roberta
    python -m scripts.features.semantic.finetune_mental_roberta --epochs 5 --batch 16 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from scripts import config
from scripts.evaluation.metrics import save_confusion_matrix_artifacts
from scripts.features.semantic.mental_roberta import MentalRobertaExtractor
from scripts.utils.logging_utils import setup_logging
from scripts.utils.outputs import evaluation_dir, log_dir

logger = __import__("logging").getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MentalHealthDataset(Dataset):
    """Returns tokenised text and its integer class label."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.texts = df[config.TEXT_COL].fillna("").astype(str).tolist()
        self.labels = df[config.LABEL_COL].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

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
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += outputs.loss.item()
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = float((np.array(all_preds) == np.array(all_labels)).mean())
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, list, list]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

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


# ---------------------------------------------------------------------------
# CLS extraction with fine-tuned backbone
# ---------------------------------------------------------------------------


def extract_and_save_cls(split: str, extractor: MentalRobertaExtractor) -> None:
    """Extract CLS embeddings for one split and write to the semantic parquet."""
    path_map = {
        "train": config.TRAIN_PATH,
        "val": config.VAL_PATH,
        "test": config.TEST_PATH,
    }
    df = pd.read_csv(path_map[split])
    texts = df[config.TEXT_COL].fillna("").astype(str).tolist()
    post_ids = [f"{split}_{i}" for i in range(len(df))]

    logger.info("Extracting CLS embeddings for %s split (%d texts)...", split, len(texts))
    matrix = extractor.extract_batch(texts)

    out_dir = config.SEMANTIC_FEATURES_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mental_roberta.parquet"
    rows = [
        {"post_id": pid, "features": vec.tolist()}
        for pid, vec in zip(post_ids, matrix)
    ]
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    logger.info("Saved %s embeddings -> %s", split, out_path)


# ---------------------------------------------------------------------------
# Main fine-tuning function
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finetune(
    epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    warmup_ratio: float = config.WARMUP_RATIO,
    grad_clip: float = config.GRAD_CLIP,
    seed: int = config.SEED,
    extract_embeddings: bool = True,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories
    eval_root = evaluation_dir("roberta", "finetune")
    log_root = log_dir("roberta", "finetune")
    checkpoint_dir = config.ROBERTA_MODEL_DIR / "checkpoints" / "best_model"
    for d in [eval_root, log_root, checkpoint_dir, config.FINETUNED_ROBERTA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    log_file = log_root / "finetune.log"
    setup_logging(log_file=log_file)

    logger.info("=" * 60)
    logger.info(
        "MentalRoBERTa fine-tuning  model=%s  device=%s",
        config.MENTAL_ROBERTA_NAME,
        device,
    )
    logger.info(
        "epochs=%d  batch=%d  lr=%g  weight_decay=%g  warmup_ratio=%g  seed=%d",
        epochs, batch_size, lr, weight_decay, warmup_ratio, seed,
    )
    logger.info("=" * 60)

    # Load data
    logger.info("Loading processed splits...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    logger.info(
        "Splits: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df)
    )

    # Tokenizer and datasets
    logger.info("Initialising tokenizer from '%s'...", config.MENTAL_ROBERTA_NAME)
    tokenizer = AutoTokenizer.from_pretrained(config.MENTAL_ROBERTA_NAME)

    train_loader = DataLoader(
        MentalHealthDataset(train_df, tokenizer, config.MAX_LENGTH),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        MentalHealthDataset(val_df, tokenizer, config.MAX_LENGTH),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        MentalHealthDataset(test_df, tokenizer, config.MAX_LENGTH),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Model
    logger.info("Loading model '%s' for %d-class classification...", config.MENTAL_ROBERTA_NAME, config.NUM_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MENTAL_ROBERTA_NAME,
        num_labels=config.NUM_LABELS,
        ignore_mismatched_sizes=True,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")

    # Optimizer and schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info("Total steps: %d  Warmup steps: %d", total_steps, warmup_steps)

    # Training loop
    history: list[dict] = []
    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device, grad_clip
        )
        va_loss, va_acc, va_f1, _, _ = eval_epoch(model, val_loader, device)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d  train: loss=%.4f acc=%.4f f1=%.4f  "
            "val: loss=%.4f acc=%.4f f1=%.4f  (%.1fs)",
            epoch, epochs, tr_loss, tr_acc, tr_f1, va_loss, va_acc, va_f1, elapsed,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 6),
                "train_acc": round(tr_acc, 6),
                "train_f1": round(tr_f1, 6),
                "val_loss": round(va_loss, 6),
                "val_acc": round(va_acc, 6),
                "val_f1": round(va_f1, 6),
            }
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            model.save_pretrained(str(checkpoint_dir))
            tokenizer.save_pretrained(str(checkpoint_dir))
            logger.info(
                "  -> New best checkpoint saved (val macro-F1=%.4f)", best_val_f1
            )

    logger.info(
        "Best val macro-F1=%.4f at epoch %d", best_val_f1, best_epoch
    )

    # Evaluate best checkpoint on test set
    logger.info("Loading best checkpoint for test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir)).to(device)
    te_loss, te_acc, te_f1, te_preds, te_labels = eval_epoch(best_model, test_loader, device)
    logger.info(
        "Test  loss=%.4f  acc=%.4f  macro-F1=%.4f", te_loss, te_acc, te_f1
    )

    class_names = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]
    report_str = classification_report(
        te_labels, te_preds, target_names=class_names, zero_division=0
    )
    logger.info("Classification report:\n%s", report_str)

    cm_artifacts = save_confusion_matrix_artifacts(
        np.array(te_labels),
        np.array(te_preds),
        class_names,
        eval_root / "confusion_matrix",
    )

    # Save fine-tuned backbone (no classification head) for embedding extraction
    logger.info("Saving fine-tuned backbone to '%s'...", config.FINETUNED_ROBERTA_DIR)
    best_model.base_model.save_pretrained(str(config.FINETUNED_ROBERTA_DIR))
    tokenizer.save_pretrained(str(config.FINETUNED_ROBERTA_DIR))
    logger.info("Backbone saved")

    # Persist metrics
    summary = {
        "model": config.MENTAL_ROBERTA_NAME,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "grad_clip": grad_clip,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 6),
        "test_loss": round(te_loss, 6),
        "test_acc": round(te_acc, 6),
        "test_macro_f1": round(te_f1, 6),
        "checkpoint": str(checkpoint_dir),
        "finetuned_backbone": str(config.FINETUNED_ROBERTA_DIR),
        "history": history,
        "confusion_matrix": cm_artifacts,
    }
    summary_path = eval_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved -> %s", summary_path)

    # CLS embedding extraction with the fine-tuned backbone
    if extract_embeddings:
        logger.info("Extracting CLS embeddings using the fine-tuned backbone...")
        extractor = MentalRobertaExtractor(model_dir=config.FINETUNED_ROBERTA_DIR)
        for split in ("train", "val", "test"):
            extract_and_save_cls(split, extractor)
        logger.info("All CLS embeddings saved to '%s'", config.SEMANTIC_FEATURES_DIR)

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune MentalRoBERTa and extract CLS embeddings."
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=config.WARMUP_RATIO)
    parser.add_argument("--grad-clip", type=float, default=config.GRAD_CLIP)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip CLS embedding extraction after fine-tuning.",
    )
    args = parser.parse_args()

    finetune(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        seed=args.seed,
        extract_embeddings=not args.no_extract,
    )
