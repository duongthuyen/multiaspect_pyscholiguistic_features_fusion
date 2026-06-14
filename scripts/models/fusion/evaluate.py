"""Evaluate the trained Gated Fusion model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from scripts import config
from scripts.evaluation.metrics import save_confusion_matrix_artifacts
from scripts.models.fusion.feature_loader import INPUT_CONFIGS, load_feature_tensors
from scripts.models.fusion.gated import build_gated_model
from scripts.models.fusion.train import load_labels
from scripts.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _variant_root(variant: str | None = None) -> Path:
    return config.RESULTS_DIR / config.GATED_FUSION_OUTPUT_DIR


def load_model_and_scaler(model_cfg: dict):
    variant = model_cfg["model"]
    root = _variant_root(variant)
    ckpt_path = root / "training" / "checkpoints" / "best.pt"
    scaler_path = root / "training" / "checkpoints" / "handcrafted_scaler.joblib"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = build_gated_model(variant, model_cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, joblib.load(scaler_path)


def load_split(input_config: str, split: str, scaler):
    semantic, affective, handcrafted, _ = load_feature_tensors(
        input_config=input_config,
        split=split,
    )
    labels = load_labels(split)
    hc_scaled = torch.from_numpy(scaler.transform(handcrafted.numpy()).astype(np.float32))
    loader = DataLoader(
        TensorDataset(semantic, affective, hc_scaled, labels),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    return loader, labels.numpy()


def evaluate(model_cfg: dict, split: str) -> dict:
    variant = model_cfg["model"]
    input_config = model_cfg.get("input_config", config.GATED_FUSION_INPUT_CONFIG)
    logger.info("=" * 60)
    logger.info("Evaluating variant=%s features=%s split=%s", variant, input_config, split)
    logger.info("=" * 60)

    model, scaler = load_model_and_scaler(model_cfg)
    loader, true_labels = load_split(input_config, split, scaler)
    class_names = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]

    all_preds = []
    with torch.no_grad():
        for sem, aff, hc, _ in loader:
            logits = model(sem, aff, hc)
            all_preds.append(logits.argmax(dim=1).numpy())
    preds = np.concatenate(all_preds)

    acc = float((preds == true_labels).mean())
    macro_f1 = float(f1_score(true_labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(true_labels, preds, average="weighted", zero_division=0))

    report_dict = classification_report(
        true_labels,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        true_labels,
        preds,
        target_names=class_names,
        zero_division=0,
    )

    eval_root = _variant_root(variant) / "evaluation" / split
    cm_artifacts = save_confusion_matrix_artifacts(
        true_labels,
        preds,
        class_names,
        eval_root / "confusion_matrix",
    )

    logger.info("Accuracy    : %.4f  (%.2f%%)", acc, acc * 100)
    logger.info("Macro F1    : %.4f", macro_f1)
    logger.info("Weighted F1 : %.4f", weighted_f1)
    logger.info("Classification report:\n%s", report_str)

    return {
        "model_name": variant,
        "input_config": input_config,
        "split": split,
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 6),
                "recall": round(report_dict[cls]["recall"], 6),
                "f1": round(report_dict[cls]["f1-score"], 6),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in class_names
        },
        "class_names": class_names,
        "confusion_matrix": cm_artifacts,
    }


def save_evaluation(result: dict) -> None:
    eval_root = _variant_root(result["model_name"]) / "evaluation" / result["split"]
    eval_root.mkdir(parents=True, exist_ok=True)

    json_path = eval_root / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    txt_path = eval_root / "summary.txt"
    lines = [
        f"Evaluation - {result['model_name']} features={result['input_config']} split={result['split']}",
        "=" * 60,
        f"Accuracy    : {result['accuracy']:.4f}  ({result['accuracy'] * 100:.2f}%)",
        f"Macro F1    : {result['macro_f1']:.4f}",
        f"Weighted F1 : {result['weighted_f1']:.4f}",
        "",
        f"{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}",
        "-" * 52,
    ]
    for cls in result["class_names"]:
        pc = result["per_class"][cls]
        lines.append(
            f"{cls:<12} {pc['precision']:>10.4f} {pc['recall']:>8.4f} "
            f"{pc['f1']:>8.4f} {pc['support']:>9}"
        )
    lines += [
        "",
        f"CM raw CSV : {result['confusion_matrix']['raw_csv_path']}",
        f"CM plot    : {result['confusion_matrix']['plot_path']}",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Saved -> %s", json_path)
    logger.info("Saved -> %s", txt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="gated_fusion")
    parser.add_argument("--features", choices=INPUT_CONFIGS)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    overrides = {"input_config": args.features} if args.features is not None else None
    model_cfg = config.get_gated_fusion_config(args.variant, overrides)

    setup_logging(log_file=_variant_root(args.variant) / "evaluation" / "evaluate.log")
    result = evaluate(model_cfg, args.split)
    save_evaluation(result)
