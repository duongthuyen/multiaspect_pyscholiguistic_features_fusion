"""
Mental Health Fusion — unified CLI entry point.

Usage
-----
    python -m scripts.main <command> [options]

Commands
--------
    preprocess          Clean and merge raw CSVs → data/processed/
    analyze             EDA plots + statistics    → results/analysis/
    extract             Extract feature groups    → data/features/<split>/
    combine             Combine sub-features      → data/features/<group>/<split>/combined.parquet
    train               Train fusion or classical models
    evaluate            Evaluate a trained fusion variant
    analyze-features    Per-class feature statistics (scaled heatmap / raw violin)
    analyze-branches    Gate-weight analysis for a trained fusion variant
    research-report     Export report-ready research tables and findings
    report-pack         Alias for research-report
    info                Summarise combined features and validate post_id consistency
    run                 Multi-seed / cross-attention training orchestrators

Examples
--------
    python -m scripts.main preprocess
    python -m scripts.main analyze
    python -m scripts.main extract --split train --components affective
    python -m scripts.main extract --split train --force
    python -m scripts.main combine
    python -m scripts.main train fusion
    python -m scripts.main train fusion --epochs 30 --lr 1e-3
    python -m scripts.main train classical --model logistic_regression
    python -m scripts.main train classical --model xgboost --features semantic
    python -m scripts.main evaluate --split test
    python -m scripts.main analyze-features
    python -m scripts.main analyze-features --raw
    python -m scripts.main analyze-branches
    python -m scripts.main report-pack
    python -m scripts.main info --validate
    python -m scripts.main run multi-seed
    python -m scripts.main run traditional-multi-seed --svd 300
    python -m scripts.main run cross-attention
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


# =============================================================================
# COMMAND IMPLEMENTATIONS
# =============================================================================

def cmd_preprocess(args: argparse.Namespace) -> None:
    from scripts import config
    from scripts.data.preprocessing import preprocess

    setup_logging(log_file=config.PROCESSED_DIR / "preprocess.log")
    preprocess()


def cmd_analyze(args: argparse.Namespace) -> None:
    from scripts import config
    from scripts.analysis.eda import analyze

    setup_logging(log_file=config.PLOTS_DIR / "analysis" / "analyze.log")
    analyze()


def cmd_extract(args: argparse.Namespace) -> None:
    """Feature extraction — wraps FeatureOrchestrator.extract_dataset()."""
    import pandas as pd
    from pathlib import Path
    from scripts import config
    from scripts.features.orchestrator import FeatureOrchestrator

    input_path = Path(args.input)
    split_name = args.split or input_path.stem

    log_file = config.FEATURES_DIR / split_name / "extract.log"
    setup_logging(log_file=log_file)

    logger.info("Loading input CSV: %s", input_path)
    df = pd.read_csv(input_path)

    post_id_col = args.post_id_col
    if post_id_col not in df.columns:
        df = df.copy()
        df[post_id_col] = [f"{split_name}_{i}" for i in range(len(df))]
        logger.info("Generated post_ids with prefix '%s_'", split_name)

    logger.info(
        "Extracting features  split=%s  components=%s  rows=%d",
        split_name, args.components or "all", len(df),
    )
    FeatureOrchestrator().extract_dataset(
        df,
        text_col=args.text_col,
        post_id_col=post_id_col,
        components=args.components,
        force=args.force,
        split=split_name,
    )
    logger.info("Feature extraction complete")


def cmd_combine(args: argparse.Namespace) -> None:
    from scripts.features.combination import main as combine_main
    combine_main()


def cmd_train_fusion(args: argparse.Namespace) -> None:
    from scripts import config
    from scripts.training.fusion_train import train

    overrides = {
        k: getattr(args, k)
        for k in (
            "input_config", "epochs", "lr", "batch_size", "seed",
            "label_smoothing", "gate_weight_decay", "early_stopping_patience",
            "aux_weight", "diversity_weight", "projection_dim", "gate_hidden_dim",
            "handcrafted_dropout",
        )
        if getattr(args, k, None) is not None
    }
    cfg = config.get_gated_fusion_config(overrides=overrides)
    train(cfg)


def cmd_train_classical(args: argparse.Namespace) -> None:
    from scripts.training.traditional import train_classifier

    model_name = args.model
    features = args.features

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from scripts import config
        estimator = LogisticRegression(
            max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        from scripts import config
        estimator = RandomForestClassifier(
            n_estimators=config.RANDOM_FOREST_N_ESTIMATORS,
            max_depth=config.RANDOM_FOREST_MAX_DEPTH,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
    elif model_name == "support_vector_machine":
        from sklearn.svm import SVC
        from scripts import config
        estimator = SVC(
            C=config.SVM_C,
            kernel=config.SVM_KERNEL,
            class_weight="balanced",
            random_state=args.seed,
        )
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        from scripts import config
        estimator = XGBClassifier(
            n_estimators=config.XGBOOST_N_ESTIMATORS,
            max_depth=config.XGBOOST_MAX_DEPTH,
            learning_rate=config.XGBOOST_LEARNING_RATE,
            random_state=args.seed,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        logger.error("Unknown classical model: %s", model_name)
        sys.exit(1)

    train_classifier(estimator, model_name, features,
                     svd_components=args.svd, seed=args.seed)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from scripts import config
    from scripts.evaluation.fusion_evaluate import evaluate, save_evaluation
    from scripts.utils.logging_utils import setup_logging as _setup

    overrides = {"input_config": args.features} if args.features else None
    model_cfg = config.get_gated_fusion_config(overrides=overrides)

    gf_root = config.RESULTS_DIR / config.GATED_FUSION_OUTPUT_DIR
    _setup(log_file=gf_root / "evaluation" / "evaluate.log")

    result = evaluate(model_cfg, args.split)
    save_evaluation(result)


def cmd_analyze_features(args: argparse.Namespace) -> None:
    from scripts.analysis.feature_statistics import (
        run_feature_statistics,
        run_feature_statistics_raw,
        run_effect_size_selection,
    )
    setup_logging()
    if args.select:
        run_effect_size_selection()
    elif args.raw:
        run_feature_statistics_raw()
    else:
        run_feature_statistics()


def cmd_analyze_branches(args: argparse.Namespace) -> None:
    from scripts import config
    from scripts.analysis.branch_weights import analyze_variant, save_results
    setup_logging()
    model_cfg = config.get_gated_fusion_config()
    result = analyze_variant(model_cfg)
    save_results(result)


def cmd_research_report(args: argparse.Namespace) -> None:
    """Generate report-ready research tables from current artifacts."""
    from scripts.analysis.research_report import generate_research_report

    setup_logging()
    generate_research_report(
        feature_report=args.feature_report,
        output_dir=args.output_dir,
    )


def cmd_info(args: argparse.Namespace) -> None:
    from scripts.analysis.combined_features_info import (
        export_combined_summary,
        log_combined_summary,
        validate_combined_consistency,
    )
    setup_logging()
    log_combined_summary()
    if args.validate:
        ok = validate_combined_consistency()
        if not ok:
            sys.exit(1)
    export_combined_summary()


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    # ALL constants are inlined here — no project imports at parse time.
    # This keeps `--help` fast and avoids pulling in torch/transformers/spacy
    # just to print usage text. Keep in sync with config.py and the modules
    # listed in the comments below each constant.
    INPUT_CONFIGS = [                          # feature_loader.py :: INPUT_CONFIGS
        "semantic", "lexical", "syntactic", "structural", "affective", "fused",
        "traditional",
    ]
    DEFAULT_TEXT_COL = "text"                  # config.py :: TEXT_COL
    DEFAULT_TRAIN_CSV = "data/processed/train.csv"

    CLASSICAL_MODELS = [
        "logistic_regression",
        "random_forest",
        "support_vector_machine",
        "xgboost",
    ]

    p = argparse.ArgumentParser(
        prog="python -m scripts.main",
        description="Mental Health Fusion — unified pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", metavar="<command>", required=True)

    # ── preprocess ────────────────────────────────────────────────────────────
    sub.add_parser(
        "preprocess",
        help="Clean and merge raw CSVs → data/processed/",
    )

    # ── analyze ───────────────────────────────────────────────────────────────
    sub.add_parser(
        "analyze",
        help="EDA: class distribution, text length, per-class stats → results/analysis/",
    )

    # ── extract ───────────────────────────────────────────────────────────────
    ext = sub.add_parser(
        "extract",
        help="Extract feature groups from an input CSV → data/features/<split>/",
    )
    ext.add_argument(
        "--input",
        default=DEFAULT_TRAIN_CSV,
        help="Path to processed CSV. Default: data/processed/train.csv",
    )
    ext.add_argument(
        "--text-col",
        default=DEFAULT_TEXT_COL,
        dest="text_col",
        help=f"Text column name. Default: {DEFAULT_TEXT_COL}",
    )
    ext.add_argument(
        "--post-id-col",
        default="post_id",
        dest="post_id_col",
        help="Post-ID column. Auto-generated if missing. Default: post_id",
    )
    ext.add_argument(
        "--split",
        default=None,
        help="Split name for output sub-directory. Defaults to the CSV stem.",
    )
    ext.add_argument(
        "--components",
        default=None,
        help=(
            "Comma-separated group or sub-extractor names to run. "
            "Examples: 'affective', 'affective.vader', 'lexical,syntactic'. "
            "Omit to run all groups."
        ),
    )
    ext.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if the parquet file already exists.",
    )

    # ── combine ───────────────────────────────────────────────────────────────
    sub.add_parser(
        "combine",
        help="Combine sub-feature parquets into one per group/split → data/features/<group>/<split>/combined.parquet",
    )

    # ── train ─────────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train fusion or classical models")
    train_sub = train_p.add_subparsers(dest="model_type", metavar="<type>", required=True)

    # train fusion
    tf = train_sub.add_parser(
        "fusion",
        help="Train the Gated Fusion model (neural, PyTorch)",
    )
    tf.add_argument("--features", dest="input_config", choices=INPUT_CONFIGS,
                    help="Feature input config. Default: fused")
    tf.add_argument("--epochs",   type=int,   dest="epochs",    help="Training epochs")
    tf.add_argument("--lr",       type=float, dest="lr",        help="Learning rate")
    tf.add_argument("--batch",    type=int,   dest="batch_size", help="Batch size")
    tf.add_argument("--seed",     type=int,   dest="seed",      help="Random seed")
    tf.add_argument("--label-smoothing",         type=float, dest="label_smoothing")
    tf.add_argument("--gate-weight-decay",       type=float, dest="gate_weight_decay")
    tf.add_argument("--early-stopping-patience", type=int,   dest="early_stopping_patience")
    tf.add_argument("--aux-weight",              type=float, dest="aux_weight")
    tf.add_argument("--diversity-weight",        type=float, dest="diversity_weight")
    tf.add_argument("--projection-dim",          type=int,   dest="projection_dim")
    tf.add_argument("--gate-hidden-dim",         type=int,   dest="gate_hidden_dim")
    tf.add_argument("--handcrafted-dropout",     type=float, dest="handcrafted_dropout")

    # train classical
    tc = train_sub.add_parser(
        "classical",
        help="Train a classical sklearn/xgboost classifier",
    )
    tc.add_argument(
        "--model",
        required=True,
        choices=CLASSICAL_MODELS,
        help="Classical model to train",
    )
    tc.add_argument(
        "--features",
        default="fused",
        choices=INPUT_CONFIGS,
        help="Feature input config. Default: fused",
    )
    tc.add_argument(
        "--svd",
        type=int,
        default=None,
        help="TruncatedSVD/LSA components for the TF-IDF block (traditional only).",
    )
    tc.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    ev = sub.add_parser(
        "evaluate",
        help="Evaluate the trained Gated Fusion model on a data split",
    )
    ev.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on. Default: test",
    )
    ev.add_argument(
        "--features",
        default=None,
        choices=INPUT_CONFIGS,
        dest="features",
        help="Feature config override. Defaults to the trained config.",
    )

    # ── analyze-features ──────────────────────────────────────────────────────
    af = sub.add_parser(
        "analyze-features",
        help="Per-class feature statistics → results/plots/feature_statistics/",
    )
    af.add_argument(
        "--raw",
        action="store_true",
        help="Use raw (unscaled) values and produce violin plots instead of heatmaps",
    )
    af.add_argument(
        "--select",
        action="store_true",
        help="Run effect-size feature selection and export keep/drop report CSV",
    )

    # ── analyze-branches ──────────────────────────────────────────────────────
    sub.add_parser(
        "analyze-branches",
        help="Gate-weight analysis for the trained Gated Fusion model -> results/models/fusion/gated_fusion/evaluation/",
    )

    # ── info ──────────────────────────────────────────────────────────────────
    inf = sub.add_parser(
        "info",
        help="Summarise combined features and optionally validate post_id consistency",
    )
    inf.add_argument(
        "--validate",
        action="store_true",
        help="Also validate that post_ids are aligned across all groups",
    )

    # ── research-report / report-pack ──────────────────────────────────────
    for name in ("research-report", "report-pack"):
        rr = sub.add_parser(
            name,
            help="Export report-ready research tables and findings from current artifacts",
        )
        rr.add_argument(
            "--feature-report",
            default="results/analysis/feature_statistics/feature_selection_report.csv",
            dest="feature_report",
            help="Path to feature_selection_report.csv.",
        )
        rr.add_argument(
            "--output-dir",
            default="results/report_pack",
            dest="output_dir",
            help="Output directory. Default: results/report_pack",
        )


    # ── run (multi-seed / cross-attention orchestration) ────────────────
    rn = sub.add_parser(
        "run",
        help="Run a multi-seed / cross-attention training orchestrator",
    )
    rn.add_argument(
        "runner",
        choices=["multi-seed", "traditional-multi-seed", "cross-attention"],
        help="Which orchestrator to run",
    )
    rn.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the runner (e.g. --svd 300 --seeds 0,1,2)",
    )


    return p


# =============================================================================
# DISPATCH
# =============================================================================

def cmd_run(args: argparse.Namespace) -> None:
    """Delegate to a multi-seed / cross-attention orchestrator under scripts/models/runners/."""
    import runpy
    targets = {
        "multi-seed":             "scripts.training.runners.run_multi_seed",
        "traditional-multi-seed": "scripts.training.runners.run_traditional_multi_seed",
        "cross-attention":        "scripts.training.runners.run_cross_attention",
    }
    module = targets[args.runner]
    sys.argv = [module] + (args.extra or [])
    runpy.run_module(module, run_name="__main__")


_DISPATCH = {
    "preprocess":       cmd_preprocess,
    "analyze":          cmd_analyze,
    "extract":          cmd_extract,
    "combine":          cmd_combine,
    "analyze-features": cmd_analyze_features,
    "analyze-branches": cmd_analyze_branches,
    "evaluate":         cmd_evaluate,
    "info":             cmd_info,
    "research-report":  cmd_research_report,
    "report-pack":      cmd_research_report,
    "run":              cmd_run,
}


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        if args.model_type == "fusion":
            cmd_train_fusion(args)
        elif args.model_type == "classical":
            cmd_train_classical(args)
        else:
            parser.error(f"Unknown model type: {args.model_type}")
    else:
        handler = _DISPATCH.get(args.command)
        if handler is None:
            parser.error(f"Unknown command: {args.command}")
        handler(args)


if __name__ == "__main__":
    main()
