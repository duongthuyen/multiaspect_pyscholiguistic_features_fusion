from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
INPUT_CONFIGS = list(config.FEATURE_GROUPS) + ["fused"]


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


def load_fusion_feature_tensors(
    split: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    logger.info("Loading fusion features (split=%s)...", split or "all")
    semantic_ids, semantic = load_group_features("semantic", split=split)
    affective_ids, affective = load_group_features("affective", split=split)

    handcrafted_parts = []
    handcrafted_ids = None
    for group in ["lexical", "syntactic", "structural"]:
        ids, matrix = load_group_features(group, split=split)
        if handcrafted_ids is None:
            handcrafted_ids = ids
        elif ids != handcrafted_ids:
            raise AssertionError(f"post_id order mismatch in handcrafted group {group}")
        handcrafted_parts.append(matrix)

    if semantic_ids != affective_ids or semantic_ids != handcrafted_ids:
        raise AssertionError(
            "post_id order mismatch across semantic, affective, and handcrafted inputs"
        )

    handcrafted = np.concatenate(handcrafted_parts, axis=1).astype(np.float32)
    if semantic.shape[1] != config.SEMANTIC_DIM:
        raise ValueError(f"semantic dim {semantic.shape[1]} != {config.SEMANTIC_DIM}")
    if affective.shape[1] != config.AFFECTIVE_DIM:
        raise ValueError(f"affective dim {affective.shape[1]} != {config.AFFECTIVE_DIM}")
    if handcrafted.shape[1] != config.HANDCRAFTED_DIM:
        raise ValueError(f"handcrafted dim {handcrafted.shape[1]} != {config.HANDCRAFTED_DIM}")

    logger.info(
        "Fusion features loaded  n=%d  semantic=%s  affective=%s  handcrafted=%s",
        len(semantic_ids), semantic.shape, affective.shape, handcrafted.shape,
    )
    return (
        torch.from_numpy(semantic),
        torch.from_numpy(affective),
        torch.from_numpy(handcrafted),
        semantic_ids,
    )


def load_feature_tensors(
    input_config: str = "fused",
    split: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    Return three tensors compatible with the fusion model signature.

    Fused mode loads all feature groups. Single-group mode places the selected
    group in its natural branch and zero-fills unused branches so
    Gated fusion signatures remain unchanged.
    """
    selected = input_config.lower()
    if selected == "fused":
        return load_fusion_feature_tensors(split=split)
    if selected not in config.FEATURE_GROUPS:
        raise ValueError(f"input_config must be one of {INPUT_CONFIGS}")

    logger.info("Loading single-group features: %s (split=%s)...", selected, split or "all")
    ids, matrix = load_group_features(selected, split=split)
    n_rows = matrix.shape[0]
    semantic = np.zeros((n_rows, config.SEMANTIC_DIM), dtype=np.float32)
    affective = np.zeros((n_rows, config.AFFECTIVE_DIM), dtype=np.float32)
    handcrafted = np.zeros((n_rows, config.HANDCRAFTED_DIM), dtype=np.float32)

    if selected == "semantic":
        semantic = matrix
    elif selected == "affective":
        affective = matrix
    else:
        offsets = {
            "lexical": 0,
            "syntactic": config.LEXICAL_DIM,
            "structural": config.LEXICAL_DIM + config.SYNTACTIC_DIM,
        }
        start = offsets[selected]
        end = start + config.FEATURE_DIMS[selected]
        handcrafted[:, start:end] = matrix

    return (
        torch.from_numpy(semantic),
        torch.from_numpy(affective),
        torch.from_numpy(handcrafted),
        ids,
    )


def apply_feature_masks(
    affective: torch.Tensor,
    handcrafted: torch.Tensor,
    masks: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-mask selected feature dimensions in the affective and handcrafted
    tensors according to a selection report mask dict.

    This is the standard zero-masking approach for ablation studies: the model
    architecture is UNCHANGED (same input dimensions), but features with
    mask=0 are forced to zero at both train and test time.  Results trained
    with and without the mask are directly comparable.

    Parameters
    ----------
    affective   : Tensor (N, AFFECTIVE_DIM)
    handcrafted : Tensor (N, HANDCRAFTED_DIM)
    masks : dict
        Output of ``feature_statistics.build_selection_masks()``.
        Expected keys: "affective" (np.ndarray float32, shape AFFECTIVE_DIM),
                       "handcrafted" (np.ndarray float32, shape HANDCRAFTED_DIM).

    Returns
    -------
    (affective_masked, handcrafted_masked) — same shape, masked in-place clone.
    """
    aff_mask = torch.from_numpy(masks["affective"])   # (AFFECTIVE_DIM,)
    hc_mask  = torch.from_numpy(masks["handcrafted"]) # (HANDCRAFTED_DIM,)

    aff_masked = affective * aff_mask.unsqueeze(0)    # broadcast over N
    hc_masked  = handcrafted * hc_mask.unsqueeze(0)

    n_aff_dropped = int((masks["affective"] == 0).sum())
    n_hc_dropped  = int((masks["handcrafted"] == 0).sum())
    logger.info(
        "Feature masks applied  affective: %d/%d dropped  handcrafted: %d/%d dropped",
        n_aff_dropped, len(masks["affective"]),
        n_hc_dropped,  len(masks["handcrafted"]),
    )
    return aff_masked, hc_masked


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
