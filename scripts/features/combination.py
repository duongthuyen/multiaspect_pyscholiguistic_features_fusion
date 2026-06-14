from __future__ import annotations

import logging

import pandas as pd

from scripts import config
from scripts.models.fusion.feature_loader import load_group_features
from scripts.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

GROUPS = ["semantic", "affective", "lexical", "syntactic", "structural"]
SPLITS = ["train", "val", "test"]


def combine_feature_group(group: str, split: str):
    post_ids, matrix = load_group_features(group, split=split)
    logger.info("%s / %s: %s", split, group, matrix.shape)

    out_dir = config.FEATURES_DIR / group / split
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "combined.parquet"
    pd.DataFrame(
        {"post_id": post_ids, "features": matrix.tolist()}
    ).to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    setup_logging()
    logger.info("Combining features for splits=%s  groups=%s", SPLITS, GROUPS)
    for split in SPLITS:
        for group in GROUPS:
            try:
                combine_feature_group(group, split)
            except Exception as exc:
                logger.error("Error combining %s/%s: %s", group, split, exc)


if __name__ == "__main__":
    main()
