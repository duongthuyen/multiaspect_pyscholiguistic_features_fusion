"""MentalRoBERTa model defaults and builder.

Training/evaluation live outside this module.  This file only defines the
LM-based model configuration and constructs the sequence-classification model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from scripts import config


MODEL_NAME = "mental_roberta"
PARADIGM = "lm_based"


@dataclass(frozen=True)
class MentalRobertaConfig:
    model: str = MODEL_NAME
    paradigm: str = PARADIGM
    pretrained_name: str = config.MENTAL_ROBERTA_NAME
    num_labels: int = config.NUM_LABELS
    max_length: int = config.MAX_LENGTH
    batch_size: int = config.ROBERTA_BATCH_SIZE
    learning_rate: float = config.LEARNING_RATE
    num_epochs: int = config.NUM_EPOCHS
    weight_decay: float = config.WEIGHT_DECAY
    warmup_ratio: float = config.WARMUP_RATIO
    grad_clip: float = config.GRAD_CLIP
    seed: int = config.SEED
    model_dir: Path = config.ROBERTA_MODEL_DIR
    checkpoint_dir: Path = config.ROBERTA_MODEL_DIR / "checkpoints" / "best_model"
    finetuned_backbone_dir: Path = config.FINETUNED_ROBERTA_DIR


def get_mental_roberta_config(overrides: dict | None = None) -> dict:
    """Return MentalRoBERTa defaults with optional overrides."""
    cfg = asdict(MentalRobertaConfig())
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def build_sequence_classifier(model_cfg: dict | None = None):
    """Build MentalRoBERTa with a sequence-classification head."""
    from transformers import AutoModelForSequenceClassification

    cfg = get_mental_roberta_config(model_cfg)
    return AutoModelForSequenceClassification.from_pretrained(
        cfg["pretrained_name"],
        num_labels=int(cfg["num_labels"]),
        ignore_mismatched_sizes=True,
    )
