"""
Training loop for improved gated fusion variants (v2).

Usage — YAML config (recommended):
    python -m scripts.models.train_fused_v2 --config configs/fused_v2_content_gate.yaml
    python -m scripts.models.train_fused_v2 --config configs/fused_v2_load_balance.yaml

Usage — CLI overrides (all keys can override YAML):
    python -m scripts.models.train_fused_v2 --model content_gate --epochs 20 --lr 5e-4
    python -m scripts.models.train_fused_v2 \\
        --config configs/fused_v2_load_balance.yaml --lb_weight 0.05

Supported variants:  content_gate | class_aware | load_balance | per_class_gate

Training improvements vs original train_fusion.py:
  early_stopping_patience=2   Stop if val_loss doesn't improve for 2 epochs.
  label_smoothing=0.1         Applied to all CE losses (main + aux where used).
  handcrafted_dropout=0.4     Passed to model constructor (vs 0.0 in original).
  gate_weight_decay=1e-4      Gate parameters get dedicated weight decay; other
                               parameters have weight_decay=0.

Outputs — results/fused_v2/<variant>/:
  training/checkpoints/best.pt
  training/logs/train.log
  training/logs/history.json
  training/logs/gate_stats_per_epoch.jsonl   per-epoch val gate weights
  training/training_history.csv
  evaluation/summary.json
  evaluation/summary.txt
  evaluation/gate_weights_per_class.csv
  evaluation/gate_weight_range.json
  evaluation/classification_report.csv
  evaluation/confusion_matrix/  (raw.csv, raw.json, plot.png)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts import config
from scripts.evaluation.metrics import save_confusion_matrix_artifacts
from scripts.models.fusion.feature_loader import INPUT_CONFIGS, load_feature_tensors
from scripts.models.fusion.gated_fusion_v2 import build_v2_model
from scripts.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# Baseline gate weight ranges from the original collapsed gated fusion.
_BASELINE_RANGES_PP = {"semantic": 3.1, "affective": 1.4, "handcrafted": 2.2}
_ROBERTA_BASELINE_F1 = 0.8873


# ---------------------------------------------------------------------------
# Path helpers — isolated from original outputs.py
# ---------------------------------------------------------------------------

def _v2_root(model_name: str) -> Path:
    return config.RESULTS_DIR / "fused_v2" / model_name


def _v2_ckpt_dir(model_name: str) -> Path:
    return _v2_root(model_name) / "training" / "checkpoints"


def _v2_log_dir(model_name: str) -> Path:
    return _v2_root(model_name) / "training" / "logs"


def _v2_eval_dir(model_name: str) -> Path:
    return _v2_root(model_name) / "evaluation"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_labels(split: str) -> torch.Tensor:
    path_map = {
        "train": config.TRAIN_PATH,
        "val": config.VAL_PATH,
        "test": config.TEST_PATH,
    }
    df = pd.read_csv(path_map[split])
    return torch.tensor(df[config.LABEL_COL].values, dtype=torch.long)


def _load_split(split: str, input_config: str):
    semantic, affective, handcrafted, _ = load_feature_tensors(
        input_config=input_config, split=split
    )
    labels = _load_labels(split)
    if len(semantic) != len(labels):
        raise AssertionError(
            f"{split}: feature count {len(semantic)} != label count {len(labels)}"
        )
    return semantic, affective, handcrafted, labels


def _scale_hc(scaler: StandardScaler, tensor: torch.Tensor, fit: bool = False) -> torch.Tensor:
    arr = tensor.numpy()
    scaled = scaler.fit_transform(arr) if fit else scaler.transform(arr)
    return torch.from_numpy(scaled.astype(np.float32))


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Optimizer — gate params get dedicated weight decay
# ---------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, lr: float, gate_wd: float = 1e-4) -> torch.optim.Adam:
    gate_ids = {id(p) for p in model.gate_parameters()}
    gate_params = [p for p in model.parameters() if id(p) in gate_ids]
    other_params = [p for p in model.parameters() if id(p) not in gate_ids]
    return torch.optim.Adam([
        {"params": gate_params, "weight_decay": gate_wd, "lr": lr},
        {"params": other_params, "weight_decay": 0.0, "lr": lr},
    ])


# ---------------------------------------------------------------------------
# Training / eval loop
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Single epoch — training if optimizer provided, else eval."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for sem, aff, hc, labels in loader:
            sem = sem.to(device)
            aff = aff.to(device)
            hc = hc.to(device)
            labels = labels.to(device)

            if training:
                total_loss_t, logits, _ = model.training_step(sem, aff, hc, labels, criterion)
            else:
                logits = model(sem, aff, hc)
                total_loss_t = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                total_loss_t.backward()
                optimizer.step()

            total_loss += total_loss_t.item()
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


# ---------------------------------------------------------------------------
# Gate statistics collection
# ---------------------------------------------------------------------------

def collect_gate_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """
    Run the model with return_gates=True on a split and return per-class
    mean gate weights plus overall mean/std.

    Returns:
        {
          "per_class_mean":  {cls_name: [sem_w, aff_w, hc_w], ...},
          "overall_mean":    [sem_w, aff_w, hc_w],
          "overall_std":     [sem_s, aff_s, hc_s],
        }
    """
    model.eval()
    all_gates: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for sem, aff, hc, labels in loader:
            _, gates = model(sem.to(device), aff.to(device), hc.to(device), return_gates=True)
            all_gates.append(gates.cpu())
            all_labels.append(labels)
    gates_np = torch.cat(all_gates).numpy()    # (N, 3)
    labels_np = torch.cat(all_labels).numpy()  # (N,)

    per_class_mean = {}
    for cls_id, cls_name in enumerate(class_names):
        mask = labels_np == cls_id
        if mask.sum() > 0:
            per_class_mean[cls_name] = gates_np[mask].mean(axis=0).tolist()

    return {
        "per_class_mean": per_class_mean,
        "overall_mean": gates_np.mean(axis=0).tolist(),
        "overall_std": gates_np.std(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Artifact saving helpers
# ---------------------------------------------------------------------------

def _save_gate_csv(gate_stats: dict, class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    pclass = gate_stats["per_class_mean"]
    for cls in class_names:
        w = pclass.get(cls, [None, None, None])
        rows.append({
            "class": cls,
            "semantic": f"{w[0]:.6f}" if w[0] is not None else "",
            "affective": f"{w[1]:.6f}" if w[1] is not None else "",
            "handcrafted": f"{w[2]:.6f}" if w[2] is not None else "",
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "semantic", "affective", "handcrafted"])
        writer.writeheader()
        writer.writerows(rows)


def _save_gate_range_json(gate_stats: dict, class_names: list[str], path: Path) -> dict:
    pclass = gate_stats["per_class_mean"]
    vals = {
        "semantic": [pclass[c][0] for c in class_names if c in pclass],
        "affective": [pclass[c][1] for c in class_names if c in pclass],
        "handcrafted": [pclass[c][2] for c in class_names if c in pclass],
    }
    ranges_pp = {
        branch: round((max(v) - min(v)) * 100, 2)
        for branch, v in vals.items()
        if v
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"ranges_pp": ranges_pp, "baseline_pp": _BASELINE_RANGES_PP}, f, indent=2)
    return ranges_pp


def _save_classification_report_csv(
    true_labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    report = classification_report(
        true_labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class", "precision", "recall", "f1_score", "support"]
        )
        writer.writeheader()
        for cls in class_names:
            r = report[cls]
            writer.writerow({
                "class": cls,
                "precision": f"{r['precision']:.6f}",
                "recall": f"{r['recall']:.6f}",
                "f1_score": f"{r['f1-score']:.6f}",
                "support": int(r["support"]),
            })


def _save_history_csv(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


# ---------------------------------------------------------------------------
# Stdout summary report
# ---------------------------------------------------------------------------

def print_summary_report(
    variant_name: str,
    test_acc: float,
    macro_f1: float,
    per_class_f1: np.ndarray,
    gate_stats: dict,
    class_names: list[str],
) -> None:
    """Print the standardised comparison block to stdout."""
    pclass = gate_stats["per_class_mean"]

    sem_vals = [pclass[c][0] for c in class_names if c in pclass]
    aff_vals = [pclass[c][1] for c in class_names if c in pclass]
    hc_vals = [pclass[c][2] for c in class_names if c in pclass]

    sem_rng = (max(sem_vals) - min(sem_vals)) * 100 if sem_vals else 0.0
    aff_rng = (max(aff_vals) - min(aff_vals)) * 100 if aff_vals else 0.0
    hc_rng = (max(hc_vals) - min(hc_vals)) * 100 if hc_vals else 0.0

    hc_avg = float(np.mean(hc_vals)) if hc_vals else 0.0
    aff_avg = float(np.mean(aff_vals)) if aff_vals else 0.0

    ptsd_hc = pclass.get("PTSD", [None, None, None])[2]
    anxiety_aff = pclass.get("Anxiety", [None, None, None])[1]

    any_rng_gt10 = any(r > 10.0 for r in [sem_rng, aff_rng, hc_rng])
    ptsd_above = ptsd_hc is not None and ptsd_hc > hc_avg
    anxiety_above = anxiety_aff is not None and anxiety_aff > aff_avg
    f1_ok = macro_f1 >= _ROBERTA_BASELINE_F1

    f1_parts = " ".join(
        f"{cls}={per_class_f1[i]:.4f}" for i, cls in enumerate(class_names)
    )
    hc_parts = " ".join(
        f"{cls}={pclass.get(cls, [0, 0, 0])[2]:.4f}" for cls in class_names
    )

    lines = [
        "",
        f"=== VARIANT: {variant_name} ===",
        f"Test accuracy: {test_acc:.4f}",
        f"Test macro F1: {macro_f1:.4f}",
        f"Per-class F1: {f1_parts}",
        "Gate weight ranges (max - min across classes):",
        f"  semantic:    {sem_rng:.1f} pp  (baseline: {_BASELINE_RANGES_PP['semantic']} pp)",
        f"  affective:   {aff_rng:.1f} pp  (baseline: {_BASELINE_RANGES_PP['affective']} pp)",
        f"  handcrafted: {hc_rng:.1f} pp  (baseline: {_BASELINE_RANGES_PP['handcrafted']} pp)",
        "Per-class handcrafted weights (key diagnostic):",
        f"  {hc_parts}",
        "Success criteria:",
        f"  [{'x' if any_rng_gt10 else ' '}] Any branch range > 10 pp (vs current 3 pp)",
        f"  [{'x' if ptsd_above else ' '}] PTSD handcrafted weight is now ABOVE average (currently below)",
        f"  [{'x' if anxiety_above else ' '}] Anxiety affective weight is now ABOVE average (currently below)",
        f"  [{'x' if f1_ok else ' '}] Macro F1 >= {_ROBERTA_BASELINE_F1} (RoBERTa-only baseline)",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict) -> dict:
    variant: str = cfg["model"]
    input_config: str = cfg.get("input_config", "fused")
    epochs: int = int(cfg.get("epochs", config.FUSION_EPOCHS))
    lr: float = float(cfg.get("lr", config.FUSION_LR))
    batch_size: int = int(cfg.get("batch_size", config.BATCH_SIZE))
    seed: int = int(cfg.get("seed", config.SEED))
    label_smoothing: float = float(cfg.get("label_smoothing", 0.1))
    gate_weight_decay: float = float(cfg.get("gate_weight_decay", 1e-4))
    early_stopping_patience: int = int(cfg.get("early_stopping_patience", 2))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_root = _v2_ckpt_dir(variant)
    logs_root = _v2_log_dir(variant)
    eval_root = _v2_eval_dir(variant)
    for p in [ckpt_root, logs_root, eval_root]:
        p.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=logs_root / "train.log")
    logger.info("=" * 65)
    logger.info(
        "V2 training  variant=%s  features=%s  device=%s", variant, input_config, device
    )
    logger.info(
        "epochs=%d  lr=%g  batch=%d  seed=%d  label_smoothing=%g  gate_wd=%g",
        epochs, lr, batch_size, seed, label_smoothing, gate_weight_decay,
    )
    logger.info(
        "early_stopping_patience=%d", early_stopping_patience
    )
    logger.info("=" * 65)

    logger.info("Loading features...")
    t0 = time.time()
    sem_tr, aff_tr, hc_tr, lbl_tr = _load_split("train", input_config)
    sem_va, aff_va, hc_va, lbl_va = _load_split("val", input_config)
    sem_te, aff_te, hc_te, lbl_te = _load_split("test", input_config)
    logger.info(
        "Loaded  train=%d  val=%d  test=%d  (%.1fs)",
        len(lbl_tr), len(lbl_va), len(lbl_te), time.time() - t0,
    )

    scaler = StandardScaler()
    hc_tr = _scale_hc(scaler, hc_tr, fit=True)
    hc_va = _scale_hc(scaler, hc_va)
    hc_te = _scale_hc(scaler, hc_te)

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

    model = build_v2_model(variant, cfg).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %s  params=%s / %s",
        model.__class__.__name__, f"{n_trainable:,}", f"{n_total:,}",
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = _build_optimizer(model, lr, gate_wd=gate_weight_decay)

    class_names = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]
    history: list[dict] = []
    epoch_gate_stats: list[dict] = []
    gate_stats_path = logs_root / "gate_stats_per_epoch.jsonl"

    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    best_epoch = 0
    best_state = None
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)

        logger.info(
            "Epoch %d/%d  train: loss=%.4f acc=%.4f  val: loss=%.4f acc=%.4f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )

        # Collect gate stats on val set (per-epoch).
        gs = collect_gate_stats(model, val_loader, device, class_names)
        gs_record = {"epoch": epoch, **gs}
        epoch_gate_stats.append(gs_record)
        with open(gate_stats_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(gs_record) + "\n")

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        history.append(row)

        # Model selection: best val_acc.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  -> New best val acc=%.4f at epoch %d", best_val_acc, best_epoch)

        # Early stopping: monitor val_loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (val_loss no improvement for %d epochs)",
                    epoch, early_stopping_patience,
                )
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    logger.info(
        "Best val acc=%.4f at epoch %d — evaluating on test...",
        best_val_acc, best_epoch,
    )
    model.load_state_dict(best_state)
    test_loss, test_acc = _run_epoch(model, test_loader, criterion, None, device)
    preds, true_labels = _predict(model, test_loader, device)

    macro_f1 = float(f1_score(true_labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(true_labels, preds, average="weighted", zero_division=0))
    per_class_f1 = f1_score(true_labels, preds, average=None, zero_division=0)

    report_str = classification_report(
        true_labels, preds, target_names=class_names, zero_division=0
    )
    logger.info("Test  loss=%.4f  acc=%.4f  macro_f1=%.4f", test_loss, test_acc, macro_f1)
    logger.info("Classification report:\n%s", report_str)

    # Gate analysis on test set.
    test_gate_stats = collect_gate_stats(model, test_loader, device, class_names)

    per_class_acc = {}
    for cls_id, cls_name in config.ID_TO_CLASS.items():
        mask = true_labels == cls_id
        per_class_acc[cls_name] = (
            round(float((preds[mask] == cls_id).mean()), 6) if mask.sum() > 0 else None
        )

    ckpt_path = ckpt_root / "best.pt"
    scaler_path = ckpt_root / "handcrafted_scaler.joblib"
    torch.save(best_state, ckpt_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Checkpoint saved -> %s", ckpt_path)

    cm_artifacts = save_confusion_matrix_artifacts(
        true_labels, preds, class_names, eval_root / "confusion_matrix"
    )

    # Save analysis artifacts.
    gate_csv_path = eval_root / "gate_weights_per_class.csv"
    gate_range_path = eval_root / "gate_weight_range.json"
    clf_report_path = eval_root / "classification_report.csv"
    history_csv_path = logs_root.parent / "training_history.csv"

    _save_gate_csv(test_gate_stats, class_names, gate_csv_path)
    ranges_pp = _save_gate_range_json(test_gate_stats, class_names, gate_range_path)
    _save_classification_report_csv(true_labels, preds, class_names, clf_report_path)
    _save_history_csv(history, history_csv_path)

    results = {
        "model_name": variant,
        "model_class": model.__class__.__name__,
        "input_config": input_config,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "label_smoothing": label_smoothing,
        "early_stopping_patience": early_stopping_patience,
        "best_val_acc": round(best_val_acc, 6),
        "best_val_loss": round(best_val_loss, 6),
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class_f1": {
            cls: round(float(per_class_f1[i]), 6) for i, cls in enumerate(class_names)
        },
        "per_class_acc": per_class_acc,
        "gate_stats_test": test_gate_stats,
        "gate_ranges_pp": ranges_pp,
        "history": history,
        "checkpoint_path": str(ckpt_path),
        "scaler_path": str(scaler_path),
        "confusion_matrix": cm_artifacts,
    }
    _save_results(results, eval_root, logs_root)

    # Stdout summary for quick comparison across runs.
    print_summary_report(variant, test_acc, macro_f1, per_class_f1, test_gate_stats, class_names)

    return results


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _save_results(results: dict, eval_root: Path, logs_root: Path) -> None:
    eval_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    json_path = eval_root / "summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    history_path = logs_root / "history.json"
    with open(history_path, "w") as f:
        json.dump(results["history"], f, indent=2)

    txt_path = eval_root / "summary.txt"
    lines = [
        f"V2 Fusion Summary — {results['model_name']} ({results['model_class']})",
        "=" * 60,
        f"Input config          : {results['input_config']}",
        f"Epochs trained        : {results['epochs_trained']}  (best: {results['best_epoch']})",
        f"Learning rate         : {results['lr']}",
        f"Batch size            : {results['batch_size']}",
        f"Seed                  : {results['seed']}",
        f"Label smoothing       : {results['label_smoothing']}",
        f"Early stop patience   : {results['early_stopping_patience']}",
        f"Best val acc          : {results['best_val_acc']:.4f}",
        f"Test accuracy         : {results['test_acc']:.4f}",
        f"Test macro F1         : {results['macro_f1']:.4f}",
        f"Test weighted F1      : {results['weighted_f1']:.4f}",
        "",
        "Per-class F1 (test):",
    ]
    for cls, f1 in results["per_class_f1"].items():
        lines.append(f"  {cls:<12} {f1:.4f}")
    lines += [
        "",
        "Per-class accuracy (test):",
    ]
    for cls, acc in results["per_class_acc"].items():
        lines.append(f"  {cls:<12} {f'{acc:.4f}' if acc is not None else 'N/A'}")
    lines += [
        "",
        "Gate weight ranges (test, pp = percentage points):",
    ]
    for branch, rng in results["gate_ranges_pp"].items():
        base = _BASELINE_RANGES_PP.get(branch, "?")
        lines.append(f"  {branch:<12} {rng:.1f} pp  (baseline: {base} pp)")
    lines += [
        "",
        f"Checkpoint      : {results['checkpoint_path']}",
        f"CM plot         : {results['confusion_matrix']['plot_path']}",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Summary JSON -> %s", json_path)
    logger.info("Summary TXT  -> %s", txt_path)
    logger.info("History JSON -> %s", history_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train improved gated fusion variants (v2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", help="YAML config file path (CLI flags override any key).")
    p.add_argument(
        "--model",
        choices=["content_gate", "class_aware", "load_balance", "per_class_gate"],
        help="Variant to train.",
    )
    p.add_argument("--features", dest="input_config", choices=INPUT_CONFIGS)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch", type=int, dest="batch_size")
    p.add_argument("--seed", type=int)
    p.add_argument("--label_smoothing", type=float)
    p.add_argument("--gate_weight_decay", type=float)
    p.add_argument("--early_stopping_patience", type=int)
    p.add_argument("--aux_weight", type=float)
    p.add_argument("--lb_weight", type=float)
    p.add_argument("--projection_dim", type=int)
    p.add_argument("--gate_hidden_dim", type=int)
    p.add_argument("--handcrafted_dropout", type=float)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg: dict = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides — only set if the arg was explicitly provided (not None).
    for key in (
        "model", "input_config", "epochs", "lr", "batch_size", "seed",
        "label_smoothing", "gate_weight_decay", "early_stopping_patience",
        "aux_weight", "lb_weight", "projection_dim", "gate_hidden_dim",
        "handcrafted_dropout",
    ):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    if "model" not in cfg:
        raise SystemExit(
            "error: --model is required (or provide a YAML config with a 'model' key)"
        )

    train(cfg)
