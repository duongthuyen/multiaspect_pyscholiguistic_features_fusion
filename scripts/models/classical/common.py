from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import config
from scripts.evaluation.metrics import save_confusion_matrix_artifacts
from scripts.models.fusion.feature_loader import INPUT_CONFIGS, load_flat_feature_matrix
from scripts.models.train_fusion import load_labels
from scripts.utils.logging_utils import setup_logging
from scripts.utils.outputs import checkpoint_dir, evaluation_dir, log_dir

logger = logging.getLogger(__name__)


def load_xy(input_config: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    _, features = load_flat_feature_matrix(input_config=input_config, split=split)
    labels = load_labels(split).numpy()
    if len(features) != len(labels):
        raise AssertionError(
            f"{split}: feature count {len(features)} != label count {len(labels)}"
        )
    return features, labels


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    input_config: str,
    model_name: str,
    split: str,
) -> dict:
    class_names = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]
    acc = float((y_pred == y_true).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm_artifacts = save_confusion_matrix_artifacts(
        y_true,
        y_pred,
        class_names,
        evaluation_dir(input_config, model_name) / split / "confusion_matrix",
    )

    return {
        "model_type": model_name,
        "input_config": input_config,
        "split": split,
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 6),
                "recall": round(report[cls]["recall"], 6),
                "f1": round(report[cls]["f1-score"], 6),
                "support": int(report[cls]["support"]),
            }
            for cls in class_names
        },
        "class_names": class_names,
        "confusion_matrix": cm_artifacts,
    }


def save_metrics(result: dict, input_config: str, model_name: str, split: str) -> None:
    eval_root = evaluation_dir(input_config, model_name) / split
    eval_root.mkdir(parents=True, exist_ok=True)

    json_path = eval_root / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    txt_path = eval_root / "summary.txt"
    lines = [
        f"Evaluation - {model_name}  features={input_config}  split={split}",
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


def train_classifier(estimator, model_name: str, input_config: str = "fused") -> dict:
    logs_root = log_dir(input_config, model_name)
    logs_root.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=logs_root / "train.log")

    logger.info("Training %s  features=%s", model_name, input_config)

    x_train, y_train = load_xy(input_config, "train")
    x_val, y_val = load_xy(input_config, "val")
    x_test, y_test = load_xy(input_config, "test")
    logger.info(
        "Data loaded  train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test)
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )
    logger.info("Fitting pipeline...")
    pipeline.fit(x_train, y_train)
    logger.info("Fitting complete")

    val_result = evaluate_predictions(
        y_val, pipeline.predict(x_val), input_config, model_name, "val"
    )
    test_result = evaluate_predictions(
        y_test, pipeline.predict(x_test), input_config, model_name, "test"
    )

    ckpt_root = checkpoint_dir(input_config, model_name)
    logs_root = log_dir(input_config, model_name)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    model_path = ckpt_root / "model.joblib"
    joblib.dump(pipeline, model_path)

    run_summary = {
        "model_type": model_name,
        "input_config": input_config,
        "model_path": str(model_path),
        "train_samples": int(len(y_train)),
        "val": val_result,
        "test": test_result,
    }
    with open(logs_root / "training_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    save_metrics(val_result, input_config, model_name, "val")
    save_metrics(test_result, input_config, model_name, "test")

    logger.info("Validation accuracy : %.4f", val_result["accuracy"])
    logger.info("Test accuracy       : %.4f", test_result["accuracy"])
    logger.info("Model saved         -> %s", model_path)
    return run_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="fused", choices=INPUT_CONFIGS)
    return parser
