"""Torch-free data loading for the *traditional* (non-neural) paradigm.

The traditional paradigm represents each post with two complementary blocks:

  * raw text          -> TF-IDF (sparse) : content / topic signal
  * dense features    -> handcrafted (lexical+syntactic+structural) + affective :
                         style / emotion signal  (NO semantic embeddings —
                         those belong to the LM-based paradigm)

This module deliberately avoids the deep-learning stack. It reuses the
numpy-only group loaders from ``feature_loader`` (which no longer import torch
at module level) and reads text / labels straight from the processed CSVs, so
the whole traditional pipeline runs without torch installed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts import config
from scripts.data.feature_loader import load_group_features

logger = logging.getLogger(__name__)

# Dense groups that make up the traditional feature block (semantic excluded).
TRADITIONAL_DENSE_GROUPS = ["lexical", "syntactic", "structural", "affective"]

# Column name used for the raw-text column inside the combined DataFrame.
TEXT_COLUMN = "tfidf_text"

_SPLIT_PATHS = {
    "train": config.TRAIN_PATH,
    "val": config.VAL_PATH,
    "test": config.TEST_PATH,
}


def _read_split_csv(split: str) -> pd.DataFrame:
    if split not in _SPLIT_PATHS:
        raise ValueError(f"Unknown split: {split!r} (expected train/val/test)")
    return pd.read_csv(_SPLIT_PATHS[split])


def load_labels_np(split: str) -> np.ndarray:
    """Integer class labels for a split.

    Returns
    -------
    np.ndarray of shape (N,), dtype int — read from the ``class_id`` column.
    """
    return _read_split_csv(split)[config.LABEL_COL].to_numpy()


def load_texts(split: str) -> list[str]:
    """Raw post texts for a split (list[str], length N)."""
    return _read_split_csv(split)[config.TEXT_COL].astype(str).tolist()


def load_dense_block(split: str) -> tuple[list, np.ndarray]:
    """Concatenate the dense (non-semantic) feature groups.

    Returns
    -------
    (post_ids, X) where X has shape (N, 60):
        lexical(11) + syntactic(8) + structural(7) + affective(34).
    """
    reference_ids = None
    matrices = []
    for group in TRADITIONAL_DENSE_GROUPS:
        ids, matrix = load_group_features(group, split=split)
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise AssertionError(f"post_id order mismatch in dense group {group}")
        matrices.append(matrix)
    X = np.concatenate(matrices, axis=1).astype(np.float32)
    return reference_ids or [], X


def dense_feature_columns(n_dense: int) -> list[str]:
    """Stable column names for the dense block: dense_0 .. dense_{n-1}."""
    return [f"dense_{i}" for i in range(n_dense)]


def build_traditional_frame(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the model-ready frame for one split.

    The text and the cached dense features are aligned by *row order* — the same
    contract the fusion code already relies on (features were extracted from the
    processed CSVs in order). We assert equal lengths as a guard.

    Returns
    -------
    (frame, y):
        frame : DataFrame with column TEXT_COLUMN (raw text) followed by
                dense_0..dense_{D-1} numeric columns.
        y     : np.ndarray (N,) integer labels.
    """
    texts = load_texts(split)
    y = load_labels_np(split)
    _, X_dense = load_dense_block(split)

    if not (len(texts) == len(y) == X_dense.shape[0]):
        raise AssertionError(
            f"{split}: length mismatch — text={len(texts)} "
            f"labels={len(y)} dense={X_dense.shape[0]}"
        )

    cols = dense_feature_columns(X_dense.shape[1])
    frame = pd.DataFrame(X_dense, columns=cols)
    frame.insert(0, TEXT_COLUMN, texts)
    logger.info(
        "Traditional frame [%s]: n=%d  text=1  dense=%d",
        split, len(frame), X_dense.shape[1],
    )
    return frame, y
