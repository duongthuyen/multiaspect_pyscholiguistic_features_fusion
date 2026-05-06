from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts import config
from scripts.features.orchestrator import FeatureOrchestrator
from scripts.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract configured feature components.")
    parser.add_argument("--input", default=str(config.TRAIN_PATH), help="Input CSV path.")
    parser.add_argument("--text-col", default=config.TEXT_COL)
    parser.add_argument("--post-id-col", default="post_id")
    parser.add_argument(
        "--split",
        default=None,
        help="Split name used in output filenames. Defaults to the input CSV stem.",
    )
    parser.add_argument(
        "--components",
        default=None,
        help=(
            "Group or sub-extractor selection, e.g. 'affective', "
            "'affective.vader', or 'affective.vad,affective.vader'."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet files.")
    args = parser.parse_args()

    split_name = args.split or Path(args.input).stem
    log_file = config.FEATURES_DIR / split_name / "extract.log"
    setup_logging(log_file=log_file)

    logger.info("Loading input CSV: %s", args.input)
    df = pd.read_csv(args.input)
    if args.post_id_col not in df.columns:
        df = df.copy()
        df[args.post_id_col] = [f"{split_name}_{i}" for i in range(len(df))]
        logger.info("Generated post_ids with prefix '%s_'", split_name)

    logger.info(
        "Extracting features  split=%s  components=%s  rows=%d",
        split_name, args.components or "all", len(df),
    )
    FeatureOrchestrator().extract_dataset(
        df,
        text_col=args.text_col,
        post_id_col=args.post_id_col,
        components=args.components,
        force=args.force,
        split=split_name,
    )
    logger.info("Feature extraction complete")


if __name__ == "__main__":
    main()
