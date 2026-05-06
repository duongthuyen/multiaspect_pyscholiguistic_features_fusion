"""
Evaluate trained fusion models from the feature-aware output structure.

Usage:
    python -m scripts.models.evaluate_fusion --model concat --features fused
    python -m scripts.models.evaluate_fusion --model gated --features semantic --split val
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from scripts import config
from scripts.evaluation.metrics import save_confusion_matrix_artifacts
from scripts.models.fusion.factory import build_fusion_model
from scripts.models.fusion.feature_loader import INPUT_CONFIGS, load_feature_tensors
from scripts.models.train_fusion import load_labels
from scripts.utils.logging_utils import setup_logging
from scripts.utils.outputs import checkpoint_dir, evaluation_dir, log_dir

logger = logging.getLogger(__name__)


def load_model_and_scaler(model_type: str, input_config: str):
    ckpt_path = checkpoint_dir(input_config, model_type) / "best.pt"
    scaler_path = checkpoint_dir(input_config, model_type) / "handcrafted_scaler.joblib"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = build_fusion_model(model_type)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler


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


def evaluate(model_type: str, input_config: str, split: str) -> dict:
    logger.info("=" * 60)
    logger.info(
        "Evaluating  model=%s  features=%s  split=%s", model_type, input_config, split
    )
    logger.info("=" * 60)

    model, scaler = load_model_and_scaler(model_type, input_config)
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

    eval_root = evaluation_dir(input_config, model_type) / split
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
        "model_type": model_type,
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
    eval_root = evaluation_dir(result["input_config"], result["model_type"]) / result["split"]
    eval_root.mkdir(parents=True, exist_ok=True)

    json_path = eval_root / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    txt_path = eval_root / "summary.txt"
    lines = [
        f"Evaluation - {result['model_type']}  features={result['input_config']}  split={result['split']}",
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


def print_comparison(results: list[dict]) -> None:
    if len(results) < 2:
        return
    logger.info("=" * 60)
    logger.info("Model comparison")
    logger.info("=" * 60)
    header = f"{'Features':<12} {'Model':<10} {'Accuracy':>9} {'Macro F1':>9} {'Wtd F1':>8}"
    logger.info(header)
    logger.info("-" * 58)
    for result in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        logger.info(
            "%-12s %-10s %9.4f %9.4f %8.4f",
            result["input_config"], result["model_type"],
            result["accuracy"], result["macro_f1"], result["weighted_f1"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", choices=["concat", "gated"], dest="models")
    parser.add_argument("--features", action="append", choices=INPUT_CONFIGS, dest="features")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    models = args.models or ["concat", "gated"]
    feature_configs = args.features or ["fused"]

    # Best-effort log file: use the first model/feature combo
    _logs_root = log_dir(feature_configs[0], models[0])
    setup_logging(log_file=_logs_root / "evaluate.log")

    all_results = []
    for input_config in feature_configs:
        for model_type in models:
            try:
                result = evaluate(model_type, input_config, args.split)
                save_evaluation(result)
                all_results.append(result)
            except FileNotFoundError as exc:
                logger.warning("SKIP %s/%s: %s", input_config, model_type, exc)

    print_comparison(all_results)
