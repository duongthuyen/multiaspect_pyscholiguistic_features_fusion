"""
Train LateConcatFusion or GatedFusion on either one feature group or all groups.

Usage:
    python -m scripts.models.train_fusion --model concat --features fused
    python -m scripts.models.train_fusion --model gated --features semantic
    python -m scripts.models.train_fusion --model concat --features lexical --epochs 20

Outputs are grouped by feature configuration:
    results/semantic/training/...
    results/fused/late_concat/training/...
    results/fused/gated/training/...
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts import config
from scripts.evaluation.metrics import save_confusion_matrix_artifacts
from scripts.models.fusion.factory import build_fusion_model
from scripts.models.fusion.feature_loader import INPUT_CONFIGS, load_feature_tensors
from scripts.utils.logging_utils import setup_logging
from scripts.utils.outputs import checkpoint_dir, evaluation_dir, log_dir

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(split: str) -> torch.Tensor:
    path_map = {"train": config.TRAIN_PATH, "val": config.VAL_PATH, "test": config.TEST_PATH}
    df = pd.read_csv(path_map[split])
    return torch.tensor(df[config.LABEL_COL].values, dtype=torch.long)


def _load_split_tensors(split: str, input_config: str):
    semantic, affective, handcrafted, _ = load_feature_tensors(
        input_config=input_config,
        split=split,
    )
    labels = load_labels(split)
    if len(semantic) != len(labels):
        raise AssertionError(
            f"{split}: feature count {len(semantic)} != label count {len(labels)}"
        )
    return semantic, affective, handcrafted, labels


def _scale_handcrafted(
    scaler: StandardScaler, tensor: torch.Tensor, fit: bool = False
) -> torch.Tensor:
    arr = tensor.numpy()
    scaled = scaler.fit_transform(arr) if fit else scaler.transform(arr)
    return torch.from_numpy(scaled.astype(np.float32))


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def run_epoch(
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
        for semantic, affective, handcrafted, labels in loader:
            semantic = semantic.to(device)
            affective = affective.to(device)
            handcrafted = handcrafted.to(device)
            labels = labels.to(device)

            logits = model(semantic, affective, handcrafted)
            loss_t = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss_t.backward()
                optimizer.step()

            total_loss += loss_t.item()
            total_acc += accuracy(logits.detach(), labels)
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


def train(
    model_type: str = "concat",
    input_config: str = "fused",
    epochs: int = config.FUSION_EPOCHS,
    lr: float = config.FUSION_LR,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_root = checkpoint_dir(input_config, model_type)
    logs_root = log_dir(input_config, model_type)
    eval_root = evaluation_dir(input_config, model_type)
    for path in [ckpt_root, logs_root, eval_root]:
        path.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=logs_root / "train.log")

    logger.info("=" * 60)
    logger.info(
        "Fusion training  model=%s  features=%s  device=%s",
        model_type.upper(), input_config, device,
    )
    logger.info(
        "epochs=%d  lr=%g  batch=%d  seed=%d", epochs, lr, batch_size, seed
    )
    logger.info("=" * 60)

    logger.info("Loading features...")
    t0 = time.time()
    sem_tr, aff_tr, hc_tr, lbl_tr = _load_split_tensors("train", input_config)
    sem_va, aff_va, hc_va, lbl_va = _load_split_tensors("val", input_config)
    sem_te, aff_te, hc_te, lbl_te = _load_split_tensors("test", input_config)
    logger.info(
        "Loaded  train=%d  val=%d  test=%d  (%.1fs)",
        len(lbl_tr), len(lbl_va), len(lbl_te), time.time() - t0,
    )

    scaler = StandardScaler()
    hc_tr = _scale_handcrafted(scaler, hc_tr, fit=True)
    hc_va = _scale_handcrafted(scaler, hc_va)
    hc_te = _scale_handcrafted(scaler, hc_te)

    train_loader = DataLoader(
        TensorDataset(sem_tr, aff_tr, hc_tr, lbl_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(sem_va, aff_va, hc_va, lbl_va),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(sem_te, aff_te, hc_te, lbl_te),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_fusion_model(model_type).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %s  params=%s / %s",
        model.__class__.__name__, f"{trainable:,}", f"{total:,}",
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_acc = float("-inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        logger.info(
            "Epoch %d/%d  train: loss=%.4f acc=%.4f  val: loss=%.4f acc=%.4f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  -> New best val acc=%.4f at epoch %d", best_val_acc, best_epoch)

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    logger.info(
        "Best val acc=%.4f at epoch %d — evaluating on test...", best_val_acc, best_epoch
    )
    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device)
    preds, true_labels = _predict(model, test_loader, device)
    logger.info("Test  loss=%.4f  acc=%.4f", test_loss, test_acc)

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

    class_names = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]
    cm_artifacts = save_confusion_matrix_artifacts(
        true_labels,
        preds,
        class_names,
        eval_root / "confusion_matrix",
    )

    results = {
        "model_type": model_type,
        "input_config": input_config,
        "epochs_trained": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 6),
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "per_class_acc": per_class_acc,
        "history": history,
        "checkpoint_path": str(ckpt_path),
        "scaler_path": str(scaler_path),
        "confusion_matrix": cm_artifacts,
    }
    save_results(results, input_config, model_type)
    return results


def save_results(results: dict, input_config: str, model_type: str) -> None:
    logs_root = log_dir(input_config, model_type)
    eval_root = evaluation_dir(input_config, model_type)
    logs_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    json_path = eval_root / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    history_path = logs_root / "history.json"
    with open(history_path, "w") as f:
        json.dump(results["history"], f, indent=2)

    txt_path = eval_root / "summary.txt"
    lines = [
        f"Fusion Training Summary - {results['model_type'].upper()}",
        "=" * 50,
        f"Input config    : {results['input_config']}",
        f"Epochs trained  : {results['epochs_trained']}",
        f"Learning rate   : {results['lr']}",
        f"Batch size      : {results['batch_size']}",
        f"Seed            : {results['seed']}",
        f"Best epoch      : {results['best_epoch']}",
        f"Best val acc    : {results['best_val_acc']:.4f}",
        f"Test loss       : {results['test_loss']:.4f}",
        f"Test accuracy   : {results['test_acc']:.4f}",
        "",
        "Per-class accuracy (test):",
    ]
    for name, value in results["per_class_acc"].items():
        lines.append(f"  {name:<12} {f'{value:.4f}' if value is not None else 'N/A'}")

    lines += [
        "",
        f"Checkpoint : {results['checkpoint_path']}",
        f"Scaler     : {results['scaler_path']}",
        f"CM raw CSV : {results['confusion_matrix']['raw_csv_path']}",
        f"CM plot    : {results['confusion_matrix']['plot_path']}",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Checkpoint      -> %s", results["checkpoint_path"])
    logger.info("Training log    -> %s", history_path)
    logger.info("Metrics JSON    -> %s", json_path)
    logger.info("Summary         -> %s", txt_path)
    logger.info("Confusion plot  -> %s", results["confusion_matrix"]["plot_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="concat", choices=["concat", "gated"])
    parser.add_argument("--features", default="fused", choices=INPUT_CONFIGS)
    parser.add_argument("--epochs", type=int, default=config.FUSION_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.FUSION_LR)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    train(
        model_type=args.model,
        input_config=args.features,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        seed=args.seed,
    )
