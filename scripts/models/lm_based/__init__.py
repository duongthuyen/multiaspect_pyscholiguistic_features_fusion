"""LM-based paradigm (MentalRoBERTa classifier)."""

from scripts.models.lm_based.mental_roberta import (
    MODEL_NAME as MENTAL_ROBERTA_MODEL,
    PARADIGM as LM_BASED_PARADIGM,
    MentalRobertaConfig,
    build_sequence_classifier,
    get_mental_roberta_config,
)

__all__ = [
    "LM_BASED_PARADIGM",
    "MENTAL_ROBERTA_MODEL",
    "MentalRobertaConfig",
    "build_sequence_classifier",
    "get_mental_roberta_config",
]
