from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# This module is numpy-only (torch-free): it reads feature parquet groups.
# The torch tensor builders that used to live here are now in
# scripts/data/fusion_dataset.py.

from scripts import config

logger = logging.getLogger(__name__)

GROUP_SUBFEATURES = {
    "semantic": ["mental_roberta"],
    "lexical": ["diversity", "word_rates", "pronouns", "punctuation"],
    "syntactic": ["complexity", "pos_ratios", "readability"],
    "structural": ["coherence", "tense"],
    "affective": ["goemotions", "vad", "vader"],
}

GROUP_DIRS = {
    "semantic": config.SEMANTIC_FEATURES_DIR,
    "lexical": config.LEXICAL_FEATURES_DIR,
    "syntactic": config.SYNTACTIC_FEATURES_DIR,
    "structural": config.STRUCTURAL_FEATURES_DIR,
    "affective": config.AFFECTIVE_FEATURES_DIR,
}

HANDCRAFTED_GROUPS = ["lexical", "syntactic", "structural"]
# "traditional" is handled by the traditional paradigm (TF-IDF + handcrafted +
# affective); it is a valid --features choice but is NOT loaded through the
# numpy/tensor loaders below — train_classifier dispatches it separately.
INPUT_CONFIGS = list(config.FEATURE_GROUPS) + ["fused", "traditional"]


def _features_to_matrix(series: pd.Series) -> np.ndarray:
    return np.asarray(series.tolist(), dtype=np.float32)


def _subextractor_feature_path(group: str, sub_name: str, split: str | None = None) -> Path:
    base_dir = GROUP_DIRS[group]
    if split:
        base_dir = base_dir / split

    # Fine-tuned CLS embeddings from Colab take priority over the pre-extracted file.
    if sub_name == "mental_roberta":
        cls_candidate = base_dir / "cls_embeddings.parquet"
        if cls_candidate.exists():
            return cls_candidate

    candidate = base_dir / f"{sub_name}.parquet"
    if candidate.exists():
        return candidate

    if split:
        candidate_split = base_dir / f"{sub_name}_{split}.parquet"
        if candidate_split.exists():
            return candidate_split

    return candidate


def _load_wide_format(df: pd.DataFrame) -> tuple[list, np.ndarray]:
    """Handle cls_embeddings.parquet: 768 numeric columns, no post_id/features columns."""
    matrix = df.values.astype(np.float32)
    post_ids = [str(i) for i in range(len(df))]
    return post_ids, matrix


def load_subextractor_features(
    group: str,
    sub_name: str,
    split: str | None = None,
) -> tuple[list, np.ndarray]:
    path = _subextractor_feature_path(group, sub_name, split=split)
    if not path.exists():
        raise FileNotFoundError(f"Missing feature parquet: {path}")
    logger.debug("Loading %s.%s from %s", group, sub_name, path)
    df = pd.read_parquet(path)

    # Wide format: numeric column names, no post_id/features (produced by Colab extraction).
    if "post_id" not in df.columns and "features" not in df.columns:
        return _load_wide_format(df)

    if "post_id" not in df.columns or "features" not in df.columns:
        raise ValueError(f"{path} must contain columns: post_id, features")
    return df["post_id"].tolist(), _features_to_matrix(df["features"])


def _normalize_post_ids(post_ids: list[str]) -> list[str]:
    """Normalize post_ids to remove prefixes like 'train_', 'val_', etc."""
    normalized = []
    for pid in post_ids:
        if "_" in pid:
            parts = pid.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                normalized.append(parts[-1])
            else:
                normalized.append(pid)
        else:
            normalized.append(pid)
    return normalized


def load_group_features(group: str, split: str | None = None) -> tuple[list, np.ndarray]:
    if group not in GROUP_SUBFEATURES:
        raise ValueError(f"Unknown group: {group}")

    reference_ids = None
    matrices = []
    for sub_name in GROUP_SUBFEATURES[group]:
        post_ids, matrix = load_subextractor_features(group, sub_name, split=split)
        normalized_ids = _normalize_post_ids(post_ids)
        if reference_ids is None:
            reference_ids = normalized_ids
        elif normalized_ids != reference_ids:
            raise AssertionError(
                f"post_id order mismatch in {group}.{sub_name} after normalization"
            )
        matrices.append(matrix)

    return reference_ids or [], np.concatenate(matrices, axis=1).astype(np.float32)


def load_flat_feature_matrix(
    input_config: str = "fused",
    split: str | None = None,
) -> tuple[list, np.ndarray]:
    """Return a 2D feature matrix for classical classifiers."""
    selected = input_config.lower()
    if selected not in INPUT_CONFIGS:
        raise ValueError(f"input_config must be one of {INPUT_CONFIGS}")
    if selected != "fused":
        return load_group_features(selected, split=split)

    logger.info("Loading flat fused feature matrix (split=%s)...", split or "all")
    reference_ids = None
    matrices = []
    for group in config.FEATURE_GROUPS:
        ids, matrix = load_group_features(group, split=split)
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise AssertionError(f"post_id order mismatch in fused group {group}")
        matrices.append(matrix)

    return reference_ids or [], np.concatenate(matrices, axis=1).astype(np.float32)


def load_feature_tensors(
    input_config: str = "fused",
    split: str | None = None,
):
    """Backward-compatible wrapper for fusion tensor loading.

    The tensor implementation lives in scripts.data.fusion_dataset; this lazy
    import keeps feature_loader numpy-only unless callers explicitly request
    torch tensors through the legacy API.
    """
    from scripts.data.fusion_dataset import load_feature_tensors as _load_feature_tensors

    return _load_feature_tensors(input_config=input_config, split=split)
