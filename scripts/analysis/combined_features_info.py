"""
Utility to inspect and test combined feature files.
Provides functions to load, validate, and export combined features.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import config
from scripts.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

GROUPS = ["semantic", "affective", "lexical", "syntactic", "structural"]
SPLITS = ["train", "val", "test"]


def load_combined_features(group: str, split: str) -> pd.DataFrame:
    """Load combined features for a group and split."""
    path = config.FEATURES_DIR / group / split / "combined.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Combined features not found: {path}")
    return pd.read_parquet(path)


def get_combined_info(group: str, split: str) -> dict:
    """Get information about combined features."""
    try:
        df = load_combined_features(group, split)
        return {
            "group": group,
            "split": split,
            "path": str(config.FEATURES_DIR / group / split / "combined.parquet"),
            "rows": len(df),
            "feature_dim": len(df["features"].iloc[0]) if len(df) > 0 else 0,
            "sample_post_ids": df["post_id"].tolist()[:3],
            "sample_features_shape": len(df["features"].iloc[0]) if len(df) > 0 else 0,
        }
    except Exception as exc:
        return {
            "group": group,
            "split": split,
            "error": str(exc),
        }


def log_combined_summary() -> None:
    """Log a summary of all combined features."""
    logger.info("=" * 80)
    logger.info("COMBINED FEATURES SUMMARY")
    logger.info("=" * 80)

    for split in SPLITS:
        logger.info("SPLIT: %s", split.upper())
        for group in GROUPS:
            info = get_combined_info(group, split)
            if "error" in info:
                logger.warning("  %-15s MISSING  %s", group, info["error"])
            else:
                logger.info(
                    "  %-15s  %6d rows x %4d dims  post_ids=%s",
                    group, info["rows"], info["feature_dim"], info["sample_post_ids"],
                )


def export_combined_summary() -> dict:
    """Export combined features info as JSON."""
    summary: dict = {}
    for split in SPLITS:
        summary[split] = {}
        for group in GROUPS:
            summary[split][group] = get_combined_info(group, split)

    output_path = config.FEATURES_DIR / "combined_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary exported -> %s", output_path)
    return summary


def load_all_combined_tensors(split: str) -> dict:
    """Load all combined features for a split as numpy arrays."""
    tensors: dict = {}
    for group in GROUPS:
        try:
            df = load_combined_features(group, split)
            post_ids = df["post_id"].tolist()
            features = np.asarray(df["features"].tolist(), dtype=np.float32)
            tensors[group] = {
                "post_ids": post_ids,
                "features": features,
                "shape": features.shape,
            }
        except Exception as exc:
            logger.error("Error loading %s/%s: %s", group, split, exc)
    return tensors


def validate_combined_consistency() -> bool:
    """Validate that post_ids are consistent across groups within each split."""
    logger.info("=" * 80)
    logger.info("VALIDATING POST_ID CONSISTENCY")
    logger.info("=" * 80)

    issues: list[str] = []
    for split in SPLITS:
        logger.info("Split: %s", split)
        reference_ids = None
        for group in GROUPS:
            try:
                df = load_combined_features(group, split)
                post_ids = df["post_id"].tolist()

                if reference_ids is None:
                    reference_ids = post_ids
                    logger.info("  %-15s reference: %d post_ids", group, len(post_ids))
                elif post_ids == reference_ids:
                    logger.info("  %-15s consistent", group)
                else:
                    msg = f"Mismatch in {split}/{group}: {len(post_ids)} vs {len(reference_ids)}"
                    logger.warning("  %-15s MISMATCH  %s", group, msg)
                    issues.append(msg)
            except Exception as exc:
                logger.error("  %-15s ERROR  %s", group, exc)
                issues.append(str(exc))

    if not issues:
        logger.info("All combined features have consistent post_ids")
    else:
        logger.warning("Found %d validation issues", len(issues))
        for issue in issues:
            logger.warning("  - %s", issue)

    return len(issues) == 0


if __name__ == "__main__":
    setup_logging()
    log_combined_summary()
    validate_combined_consistency()
    export_combined_summary()
