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
from scripts.evaluation.classification import evaluate_predictions, save_metrics
from scripts.data.feature_loader import INPUT_CONFIGS, load_flat_feature_matrix
from scripts.data.traditional_dataset import load_labels_np
from scripts.utils.logging_utils import setup_logging
from scripts.utils.outputs import checkpoint_dir, evaluation_dir, log_dir

logger = logging.getLogger(__name__)

# Config name for the TF-IDF + handcrafted + affective traditional baseline.
TRADITIONAL_CONFIG = "traditional"


def load_xy(input_config: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    _, features = load_flat_feature_matrix(input_config=input_config, split=split)
    labels = load_labels_np(split)
    if len(features) != len(labels):
        raise AssertionError(
            f"{split}: feature count {len(features)} != label count {len(labels)}"
        )
    return features, labels


def _persist_and_report(
    pipeline,
    x_val,
    y_val,
    x_test,
    y_test,
    n_train: int,
    input_config: str,
    model_name: str,
) -> dict:
    """Shared tail: evaluate val/test, save model + metrics, return run summary."""
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
        "train_samples": int(n_train),
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


def train_classifier(estimator, model_name: str, input_config: str = "fused",
                     svd_components: int | None = None, seed: int = 42) -> dict:
    # The traditional baseline (TF-IDF + handcrafted + affective) has a different
    # data shape (raw text + dense) and a ColumnTransformer pipeline, so it gets
    # its own training path.
    if input_config == TRADITIONAL_CONFIG:
        return train_traditional_classifier(estimator, model_name,
                                            svd_components=svd_components, seed=seed)

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

    return _persist_and_report(
        pipeline, x_val, y_val, x_test, y_test, len(y_train), input_config, model_name
    )


def train_traditional_classifier(estimator, model_name: str,
                                 svd_components: int | None = None, seed: int = 42) -> dict:
    """Train the traditional baseline: TF-IDF(text) + handcrafted + affective.

    Semantic embeddings are intentionally excluded (they define the LM-based
    paradigm). TF-IDF is fit on train only, inside the pipeline, so there is no
    leakage. Results are written to results/models/traditional/<model_name>/.
    """
    from scripts.data.traditional_dataset import build_traditional_frame
    from scripts.models.traditional.tfidf_pipeline import build_traditional_pipeline

    input_config = TRADITIONAL_CONFIG
    logs_root = log_dir(input_config, model_name)
    logs_root.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=logs_root / "train.log")

    logger.info(
        "Training %s  features=%s  (TF-IDF + handcrafted + affective, no semantic)",
        model_name, input_config,
    )

    x_train, y_train = build_traditional_frame("train")
    x_val, y_val = build_traditional_frame("val")
    x_test, y_test = build_traditional_frame("test")
    n_dense = x_train.shape[1] - 1  # all columns except the text column
    logger.info(
        "Data loaded  train=%d  val=%d  test=%d  dense=%d",
        len(y_train), len(y_val), len(y_test), n_dense,
    )

    pipeline = build_traditional_pipeline(estimator, n_dense,
                                          svd_components=svd_components, seed=seed)
    logger.info("Fitting TF-IDF pipeline (svd=%s, fit on train only)...", svd_components)
    pipeline.fit(x_train, y_train)
    logger.info("Fitting complete")

    return _persist_and_report(
        pipeline, x_val, y_val, x_test, y_test, len(y_train), input_config, model_name
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="fused", choices=INPUT_CONFIGS)
    parser.add_argument("--svd", type=int, default=None,
                        help="TruncatedSVD/LSA components for the TF-IDF block (traditional only). Omit for full sparse TF-IDF.")
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser
