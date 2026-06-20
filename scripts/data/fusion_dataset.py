"""Fusion dataset: assemble the three feature branches (semantic, affective,
handcrafted) as torch tensors for the fusion models.

Sits on top of the numpy feature loader (scripts/data/feature_loader.py): it
reads the cached feature groups and wraps them as tensors with the branch
layout the fusion models expect.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from scripts import config
from scripts.data.feature_loader import load_group_features

logger = logging.getLogger(__name__)

INPUT_CONFIGS = list(config.FEATURE_GROUPS) + ["fused"]


class FusionDataset(Dataset):
    """Torch dataset for fusion models with semantic, affective, handcrafted branches."""

    def __init__(
        self,
        semantic: torch.Tensor,
        affective: torch.Tensor,
        handcrafted: torch.Tensor,
        labels: torch.Tensor | None = None,
        post_ids: list[str] | None = None,
    ) -> None:
        n_rows = len(semantic)
        if len(affective) != n_rows or len(handcrafted) != n_rows:
            raise ValueError("semantic, affective, and handcrafted tensors must align")
        if labels is not None and len(labels) != n_rows:
            raise ValueError(f"feature count {n_rows} != label count {len(labels)}")
        if post_ids is not None and len(post_ids) != n_rows:
            raise ValueError(f"feature count {n_rows} != post_id count {len(post_ids)}")

        self.semantic = semantic.float()
        self.affective = affective.float()
        self.handcrafted = handcrafted.float()
        self.labels = labels.long() if labels is not None else None
        self.post_ids = post_ids

    def __len__(self) -> int:
        return len(self.semantic)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "semantic": self.semantic[idx],
            "affective": self.affective[idx],
            "handcrafted": self.handcrafted[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        if self.post_ids is not None:
            item["post_id"] = self.post_ids[idx]
        return item


def load_labels(split: str) -> torch.Tensor:
    """Load integer class labels for a processed split."""
    path_map = {
        "train": config.TRAIN_PATH,
        "val": config.VAL_PATH,
        "test": config.TEST_PATH,
    }
    if split not in path_map:
        raise ValueError("split must be one of: train, val, test")
    df = pd.read_csv(path_map[split])
    return torch.tensor(df[config.LABEL_COL].values, dtype=torch.long)


def load_fusion_feature_tensors(
    split: str | None = None,
):
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
):
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


def load_fusion_dataset(
    input_config: str = "fused",
    split: str = "train",
    include_labels: bool = True,
) -> FusionDataset:
    """Load a processed split as a FusionDataset."""
    semantic, affective, handcrafted, post_ids = load_feature_tensors(
        input_config=input_config,
        split=split,
    )
    labels = load_labels(split) if include_labels else None
    return FusionDataset(
        semantic=semantic,
        affective=affective,
        handcrafted=handcrafted,
        labels=labels,
        post_ids=post_ids,
    )


def load_fusion_dataloader(
    input_config: str = "fused",
    split: str = "train",
    batch_size: int = config.FUSION_BATCH_SIZE,
    shuffle: bool | None = None,
    include_labels: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for fusion model training or evaluation."""
    dataset = load_fusion_dataset(
        input_config=input_config,
        split=split,
        include_labels=include_labels,
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def apply_feature_masks(
    affective,
    handcrafted,
    masks: dict,
):
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
