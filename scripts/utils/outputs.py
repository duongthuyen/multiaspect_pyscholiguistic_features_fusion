from __future__ import annotations

from pathlib import Path

from scripts import config


MODEL_FOLDER_NAMES = {
    "logistic_regression": "logistic_regression",
    "random_forest": "random_forest",
    "support_vector_machine": "support_vector_machine",
    "xgboost": "xgboost",
}


def experiment_root(input_config: str, model_name: str | None = None) -> Path:
    selected = input_config.lower()
    model_folder = MODEL_FOLDER_NAMES.get(model_name, model_name) if model_name else None
    models_root = config.RESULTS_DIR / "models"
    if selected == "fused":
        if model_name is None:
            return models_root / "fused"
        return models_root / "fused" / model_folder
    if model_name is None:
        return models_root / selected
    return models_root / selected / model_folder


def training_dir(input_config: str, model_name: str) -> Path:
    return experiment_root(input_config, model_name) / "training"


def evaluation_dir(input_config: str, model_name: str) -> Path:
    return experiment_root(input_config, model_name) / "evaluation"


def checkpoint_dir(input_config: str, model_name: str) -> Path:
    """Trained-model artifacts live under ARTIFACTS_DIR (not results/)."""
    rel = experiment_root(input_config, model_name).relative_to(config.RESULTS_DIR)
    return config.ARTIFACTS_DIR / rel / "checkpoints"


def log_dir(input_config: str, model_name: str) -> Path:
    return training_dir(input_config, model_name) / "logs"
