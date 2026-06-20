from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from scripts import config

logger = logging.getLogger(__name__)


DEFAULT_FEATURE_REPORT = (
    config.PLOTS_DIR / "feature_statistics" / "feature_selection_report.csv"
)
DEFAULT_OUTPUT_DIR = config.RESULTS_DIR / "report_pack"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.{digits}f}"


def _load_feature_report(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature selection report not found: {path}. "
            "Run: python -m scripts.main analyze-features --select"
        )
    df = pd.read_csv(path)
    required = {"feature", "group", "sub_name", "eta2", "recommendation", "effect_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature report missing columns: {sorted(missing)}")
    return df.sort_values("eta2", ascending=False).reset_index(drop=True)


def _export_feature_selection_tables(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    feature_dir = output_dir / "01_feature_selection"
    feature_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["all_ranked"] = feature_dir / "all_features_ranked.csv"
    df.to_csv(paths["all_ranked"], index=False)

    for rec in ("DROP", "KEEP", "STRONG_KEEP"):
        key = rec.lower()
        paths[key] = feature_dir / f"features_{key}.csv"
        df[df["recommendation"] == rec].to_csv(paths[key], index=False)

    paths["selected_keep_threshold"] = feature_dir / "selected_threshold_KEEP.csv"
    df[df["recommendation"].isin(["KEEP", "STRONG_KEEP"])].to_csv(
        paths["selected_keep_threshold"], index=False
    )

    paths["selected_strong_keep_threshold"] = (
        feature_dir / "selected_threshold_STRONG_KEEP.csv"
    )
    df[df["recommendation"] == "STRONG_KEEP"].to_csv(
        paths["selected_strong_keep_threshold"], index=False
    )

    counts = (
        df.groupby(["group", "recommendation"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["STRONG_KEEP", "KEEP", "DROP"], fill_value=0)
        .reset_index()
    )
    counts["total"] = counts[["STRONG_KEEP", "KEEP", "DROP"]].sum(axis=1)
    paths["group_recommendation_counts"] = feature_dir / "group_recommendation_counts.csv"
    counts.to_csv(paths["group_recommendation_counts"], index=False)

    summary = (
        df.groupby("group")
        .agg(
            n_features=("feature", "count"),
            eta2_mean=("eta2", "mean"),
            eta2_median=("eta2", "median"),
            eta2_max=("eta2", "max"),
            top_feature=("feature", "first"),
        )
        .reset_index()
        .sort_values("eta2_max", ascending=False)
    )
    paths["group_effect_summary"] = feature_dir / "group_effect_summary.csv"
    summary.to_csv(paths["group_effect_summary"], index=False)

    return paths


def _load_mental_roberta_summary() -> tuple[dict | None, dict[str, float]]:
    model_root = config.RESULTS_DIR / "models" / "lm_based" / "mental_roberta"
    summary_path = model_root / "evaluation" / "summary.json"
    if not summary_path.exists():
        summary_path = model_root / "summary.json"

    report_path = model_root / "evaluation" / "classification_report.csv"
    if not report_path.exists():
        report_path = model_root / "classification_report.csv"

    summary = _load_summary(summary_path)
    per_class_f1: dict[str, float] = {}
    if report_path.exists():
        report_df = pd.read_csv(report_path)
        label_col = report_df.columns[0]
        for _, row in report_df.iterrows():
            label = str(row[label_col])
            if label in config.ID_TO_CLASS.values():
                per_class_f1[label] = float(row["f1-score"])
    return summary, per_class_f1


def _load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _summary_row(path: Path, run_type: str) -> dict:
    data = _load_summary(path) or {}
    feature_selection = data.get("feature_selection") or {}
    gates = data.get("gate_stats_test") or {}
    ranges = data.get("gate_ranges_pp") or {}
    overall = gates.get("overall_mean") or [None, None, None]
    variant = data.get("model_name") or path.parents[1].name

    semantic = _safe_float(overall[0])
    affective = _safe_float(overall[1])
    handcrafted = _safe_float(overall[2])
    max_gate = max(v for v in [semantic, affective, handcrafted] if v is not None)
    dominant_branch = "unknown"
    if semantic == max_gate:
        dominant_branch = "semantic"
    elif affective == max_gate:
        dominant_branch = "affective"
    elif handcrafted == max_gate:
        dominant_branch = "handcrafted"

    return {
        "run_type": run_type,
        "variant": variant,
        "threshold": feature_selection.get("threshold") or "none",
        "n_kept": feature_selection.get("n_kept"),
        "n_dropped": feature_selection.get("n_dropped"),
        "test_acc": data.get("test_acc"),
        "macro_f1": data.get("macro_f1"),
        "weighted_f1": data.get("weighted_f1"),
        "best_val_acc": data.get("best_val_acc"),
        "epochs_trained": data.get("epochs_trained"),
        "semantic_gate_mean": semantic,
        "affective_gate_mean": affective,
        "handcrafted_gate_mean": handcrafted,
        "semantic_range_pp": ranges.get("semantic"),
        "affective_range_pp": ranges.get("affective"),
        "handcrafted_range_pp": ranges.get("handcrafted"),
        "dominant_branch": dominant_branch,
        "is_semantic_dominated_80": bool(semantic is not None and semantic >= 0.8),
        "summary_path": str(path),
    }


def _collect_gated_summaries(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    baseline_root = config.RESULTS_DIR / config.GATED_FUSION_OUTPUT_DIR
    selected_root = config.RESULTS_DIR / (str(config.GATED_FUSION_OUTPUT_DIR) + "_selected")

    for path in sorted(baseline_root.glob("*/evaluation/summary.json")):
        rows.append(_summary_row(path, "baseline"))
    for path in sorted(selected_root.glob("*/evaluation/summary.json")):
        rows.append(_summary_row(path, "selected"))

    results_df = pd.DataFrame(rows)
    fusion_dir = output_dir / "02_model_comparison"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(fusion_dir / "gated_fusion_runs.csv", index=False)

    deltas = []
    if not results_df.empty:
        baseline = results_df[results_df["run_type"] == "baseline"].set_index("variant")
        selected = results_df[results_df["run_type"] == "selected"].set_index("variant")
        for variant, selected_row in selected.iterrows():
            if variant not in baseline.index:
                continue
            base_row = baseline.loc[variant]
            deltas.append(
                {
                    "variant": variant,
                    "selected_threshold": selected_row["threshold"],
                    "baseline_macro_f1": base_row["macro_f1"],
                    "selected_macro_f1": selected_row["macro_f1"],
                    "delta_macro_f1": _safe_float(selected_row["macro_f1"])
                    - _safe_float(base_row["macro_f1"]),
                    "baseline_test_acc": base_row["test_acc"],
                    "selected_test_acc": selected_row["test_acc"],
                    "delta_test_acc": _safe_float(selected_row["test_acc"])
                    - _safe_float(base_row["test_acc"]),
                    "baseline_semantic_gate": base_row["semantic_gate_mean"],
                    "selected_semantic_gate": selected_row["semantic_gate_mean"],
                    "delta_semantic_gate": _safe_float(selected_row["semantic_gate_mean"])
                    - _safe_float(base_row["semantic_gate_mean"]),
                    "selected_n_dropped": selected_row["n_dropped"],
                }
            )

    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(fusion_dir / "selected_vs_baseline_deltas.csv", index=False)
    return results_df, delta_df


def _export_model_comparisons(
    fusion_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_dir = output_dir / "02_model_comparison"
    model_dir.mkdir(parents=True, exist_ok=True)

    mental_summary, mental_per_class_f1 = _load_mental_roberta_summary()
    overview_rows = []
    if mental_summary:
        overview_rows.append(
            {
                "model_id": "mental_roberta",
                "family": "mental_roberta",
                "run_type": "standalone",
                "variant": "mental_roberta",
                "threshold": "none",
                "test_acc": mental_summary.get("test_acc")
                or mental_summary.get("test_accuracy"),
                "macro_f1": mental_summary.get("test_macro_f1"),
                "weighted_f1": None,
                "best_val_acc": None,
                "best_val_macro_f1": mental_summary.get("best_val_macro_f1"),
                "semantic_gate_mean": None,
                "affective_gate_mean": None,
                "handcrafted_gate_mean": None,
                "dominant_branch": "NA",
                "summary_path": str(
                    config.RESULTS_DIR
                    / "models"
                    / "lm_based"
                    / "mental_roberta"
                    / "evaluation"
                    / "summary.json"
                ),
            }
        )

    if not fusion_df.empty:
        for row in fusion_df.itertuples(index=False):
            overview_rows.append(
                {
                    "model_id": f"{row.run_type}_{row.variant}",
                    "family": "gated_fusion",
                    "run_type": row.run_type,
                    "variant": row.variant,
                    "threshold": row.threshold,
                    "test_acc": row.test_acc,
                    "macro_f1": row.macro_f1,
                    "weighted_f1": row.weighted_f1,
                    "best_val_acc": row.best_val_acc,
                    "best_val_macro_f1": None,
                    "semantic_gate_mean": row.semantic_gate_mean,
                    "affective_gate_mean": row.affective_gate_mean,
                    "handcrafted_gate_mean": row.handcrafted_gate_mean,
                    "dominant_branch": row.dominant_branch,
                    "summary_path": row.summary_path,
                }
            )

    overview_df = pd.DataFrame(overview_rows)
    if not overview_df.empty:
        overview_df = overview_df.sort_values("macro_f1", ascending=False)
    overview_df.to_csv(model_dir / "model_overview.csv", index=False)

    per_class_rows = []
    for cls in config.ID_TO_CLASS.values():
        per_class_rows.append(
            {
                "model_id": "mental_roberta",
                "family": "mental_roberta",
                "run_type": "standalone",
                "variant": "mental_roberta",
                "threshold": "none",
                "class": cls,
                "f1": mental_per_class_f1.get(cls),
            }
        )

    for path in sorted((config.RESULTS_DIR / config.GATED_FUSION_OUTPUT_DIR).glob("*/evaluation/summary.json")):
        data = _load_summary(path) or {}
        for cls, f1 in (data.get("per_class_f1") or {}).items():
            per_class_rows.append(
                {
                    "model_id": f"baseline_{data.get('model_name', path.parents[1].name)}",
                    "family": "gated_fusion",
                    "run_type": "baseline",
                    "variant": data.get("model_name", path.parents[1].name),
                    "threshold": "none",
                    "class": cls,
                    "f1": f1,
                }
            )

    selected_root = config.RESULTS_DIR / (str(config.GATED_FUSION_OUTPUT_DIR) + "_selected")
    for path in sorted(selected_root.glob("*/evaluation/summary.json")):
        data = _load_summary(path) or {}
        fs = data.get("feature_selection") or {}
        threshold = fs.get("threshold") or "selected"
        for cls, f1 in (data.get("per_class_f1") or {}).items():
            per_class_rows.append(
                {
                    "model_id": f"selected_{data.get('model_name', path.parents[1].name)}",
                    "family": "gated_fusion",
                    "run_type": "selected",
                    "variant": data.get("model_name", path.parents[1].name),
                    "threshold": threshold,
                    "class": cls,
                    "f1": f1,
                }
            )

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(model_dir / "per_class_f1_long.csv", index=False)
    if not per_class_df.empty:
        wide = per_class_df.pivot_table(
            index="class", columns="model_id", values="f1", aggfunc="first"
        ).reset_index()
        wide.to_csv(model_dir / "per_class_f1_wide.csv", index=False)

    # Keep this file close to the other model-comparison tables.
    delta_df.to_csv(model_dir / "selected_vs_baseline_deltas.csv", index=False)
    return overview_df, per_class_df


def _export_gate_analysis(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate_dir = output_dir / "03_gate_analysis"
    gate_dir.mkdir(parents=True, exist_ok=True)

    overview_rows = []
    per_class_rows = []

    roots = [
        ("baseline", config.RESULTS_DIR / config.GATED_FUSION_OUTPUT_DIR),
        ("selected", config.RESULTS_DIR / (str(config.GATED_FUSION_OUTPUT_DIR) + "_selected")),
    ]
    for run_type, root in roots:
        for path in sorted(root.glob("*/evaluation/summary.json")):
            data = _load_summary(path) or {}
            variant = data.get("model_name", path.parents[1].name)
            fs = data.get("feature_selection") or {}
            threshold = fs.get("threshold") or "none"
            gates = data.get("gate_stats_test") or {}
            ranges = data.get("gate_ranges_pp") or {}
            overall = gates.get("overall_mean") or [None, None, None]
            overview_rows.append(
                {
                    "model_id": f"{run_type}_{variant}",
                    "run_type": run_type,
                    "variant": variant,
                    "threshold": threshold,
                    "semantic_gate_mean": overall[0],
                    "affective_gate_mean": overall[1],
                    "handcrafted_gate_mean": overall[2],
                    "semantic_range_pp": ranges.get("semantic"),
                    "affective_range_pp": ranges.get("affective"),
                    "handcrafted_range_pp": ranges.get("handcrafted"),
                }
            )
            per_class_mean = gates.get("per_class_mean") or {}
            for cls, values in per_class_mean.items():
                per_class_rows.append(
                    {
                        "model_id": f"{run_type}_{variant}",
                        "run_type": run_type,
                        "variant": variant,
                        "threshold": threshold,
                        "class": cls,
                        "semantic": values[0],
                        "affective": values[1],
                        "handcrafted": values[2],
                    }
                )

    overview_df = pd.DataFrame(overview_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    overview_df.to_csv(gate_dir / "gate_overview.csv", index=False)
    per_class_df.to_csv(gate_dir / "gate_per_class.csv", index=False)
    return overview_df, per_class_df


def _write_findings_markdown(
    feature_df: pd.DataFrame,
    fusion_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    model_overview_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    notes_dir = output_dir / "04_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    path = notes_dir / "research_findings.md"

    counts = feature_df["recommendation"].value_counts()
    strong_n = int(counts.get("STRONG_KEEP", 0))
    keep_n = int(counts.get("KEEP", 0))
    drop_n = int(counts.get("DROP", 0))
    total = int(len(feature_df))

    top_features = feature_df.head(10)[["feature", "group", "eta2", "recommendation"]]
    top_lines = [
        f"- `{row.feature}` ({row.group}): eta2={row.eta2:.4f}, {row.recommendation}"
        for row in top_features.itertuples()
    ]

    best_line = "No gated-fusion summaries found."
    if not model_overview_df.empty:
        best = model_overview_df.sort_values("macro_f1", ascending=False).iloc[0]
        best_line = (
            f"Best current run: `{best['model_id']}` "
            f"(threshold={best['threshold']}), macro F1={_fmt(best['macro_f1'])}, "
            f"test accuracy={_fmt(best['test_acc'])}."
        )

    semantic_dom = ""
    if not fusion_df.empty:
        dom_count = int(fusion_df["is_semantic_dominated_80"].sum())
        semantic_dom = (
            f"{dom_count}/{len(fusion_df)} gated-fusion summaries have mean semantic "
            "gate >= 0.80, so current fusion behavior is semantic-dominated."
        )

    delta_lines = []
    if not delta_df.empty:
        for row in delta_df.sort_values("delta_macro_f1", ascending=False).itertuples():
            dropped = "NA" if pd.isna(row.selected_n_dropped) else str(int(row.selected_n_dropped))
            delta_lines.append(
                f"- `{row.variant}` ({row.selected_threshold}): "
                f"delta macro F1={row.delta_macro_f1:+.4f}, "
                f"delta acc={row.delta_test_acc:+.4f}, "
                f"dropped={dropped}"
            )

    text = f"""# Research Findings Pack

Generated from current local artifacts.

## Feature Selection

The effect-size report contains {total} interpretable features:

- STRONG_KEEP: {strong_n}
- KEEP: {keep_n}
- DROP: {drop_n}

Top eta-squared features:

{chr(10).join(top_lines)}

Interpretation: effect-size selection is useful here as a dataset-level linguistic audit. It identifies which interpretable dimensions vary enough across labels to justify inclusion in the fusion ablation. It should not be described as a causal proof of feature importance.

## Gated Fusion

{best_line}

{semantic_dom}

Selected-feature deltas against matching baselines:

{chr(10).join(delta_lines) if delta_lines else "- No matched selected/baseline pairs found."}

## Report Claim Suggested by Current Results

The defensible research claim is not that handcrafted features substantially improve accuracy over MentalRoBERTa. The defensible claim is:

> Multi-aspect linguistic features provide an interpretable diagnostic layer for mental-health text classification. When fused with a strong domain semantic representation, gated fusion preserves strong performance but learns semantic-dominated routing; effect-size selection reduces weak interpretable dimensions without producing a clear accuracy gain.

## Outputs

- `01_feature_selection/all_features_ranked.csv`
- `01_feature_selection/features_strong_keep.csv`
- `01_feature_selection/features_keep.csv`
- `01_feature_selection/features_drop.csv`
- `01_feature_selection/selected_threshold_KEEP.csv`
- `01_feature_selection/selected_threshold_STRONG_KEEP.csv`
- `02_model_comparison/model_overview.csv`
- `02_model_comparison/per_class_f1_wide.csv`
- `02_model_comparison/selected_vs_baseline_deltas.csv`
- `03_gate_analysis/gate_overview.csv`
- `03_gate_analysis/gate_per_class.csv`
"""
    path.write_text(text, encoding="utf-8")
    return path


def _write_index_markdown(
    output_dir: Path,
    model_overview_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    gate_overview_df: pd.DataFrame,
) -> Path:
    path = output_dir / "README.md"

    best_model = "NA"
    if not model_overview_df.empty:
        best = model_overview_df.sort_values("macro_f1", ascending=False).iloc[0]
        best_model = (
            f"`{best['model_id']}` with macro F1={_fmt(best['macro_f1'])} "
            f"and accuracy={_fmt(best['test_acc'])}"
        )

    counts = feature_df["recommendation"].value_counts()
    semantic_dom = "NA"
    if not gate_overview_df.empty:
        dominated = (gate_overview_df["semantic_gate_mean"].astype(float) >= 0.8).sum()
        semantic_dom = f"{int(dominated)}/{len(gate_overview_df)} gated-fusion runs"

    text = f"""# Report Pack

This folder gathers the comparison tables needed for the research report.

## Quick Findings

- Best run: {best_model}
- Feature selection: {int(counts.get('STRONG_KEEP', 0))} STRONG_KEEP, {int(counts.get('KEEP', 0))} KEEP, {int(counts.get('DROP', 0))} DROP.
- Semantic-dominated gates: {semantic_dom} have mean semantic gate >= 0.80.

## Folder Layout

```text
01_feature_selection/
  all_features_ranked.csv
  features_strong_keep.csv
  features_keep.csv
  features_drop.csv
  selected_threshold_KEEP.csv
  selected_threshold_STRONG_KEEP.csv
  group_recommendation_counts.csv
  group_effect_summary.csv

02_model_comparison/
  model_overview.csv
  gated_fusion_runs.csv
  per_class_f1_long.csv
  per_class_f1_wide.csv
  selected_vs_baseline_deltas.csv

03_gate_analysis/
  gate_overview.csv
  gate_per_class.csv

04_notes/
  research_findings.md
```

## Feature Analysis Artifacts

Feature analysis plots and raw per-class statistics are generated outside this
pack under `results/plots/feature_statistics_raw/`. These artifacts are computed
on the training split and provide the class-level linguistic profiles used to
interpret the selected features.

```text
results/plots/feature_statistics_raw/
  affective/
    affective_heatmap_combined.png
    goemotions_heatmap_raw.png
    vad_heatmap_raw.png
    vader_heatmap_raw.png
    affective_mean_eta2.csv
  lexical/
    lexical_heatmap_combined.png
    diversity_heatmap_raw.png
    pronouns_heatmap_raw.png
    punctuation_heatmap_raw.png
    word_rates_heatmap_raw.png
    lexical_mean_eta2.csv
  syntactic/
    syntactic_heatmap_combined.png
    complexity_heatmap_raw.png
    pos_ratios_heatmap_raw.png
    readability_heatmap_raw.png
    syntactic_mean_eta2.csv
  structural/
    structural_heatmap_combined.png
    coherence_heatmap_raw.png
    tense_heatmap_raw.png
    structural_mean_eta2.csv
```

## How To Regenerate

```bash
python -m scripts.main report-pack
```

or:

```bash
python -m scripts.main research-report --output-dir results/report_pack
```
"""
    path.write_text(text, encoding="utf-8")
    return path


def generate_research_report(
    feature_report: str | Path = DEFAULT_FEATURE_REPORT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_df = _load_feature_report(Path(feature_report))
    feature_paths = _export_feature_selection_tables(feature_df, output_path)
    fusion_df, delta_df = _collect_gated_summaries(output_path)
    model_overview_df, _ = _export_model_comparisons(fusion_df, delta_df, output_path)
    gate_overview_df, _ = _export_gate_analysis(output_path)
    findings_path = _write_findings_markdown(
        feature_df, fusion_df, delta_df, model_overview_df, output_path
    )
    index_path = _write_index_markdown(
        output_path, model_overview_df, feature_df, gate_overview_df
    )

    logger.info("Research report artifacts written to %s", output_path)
    return {
        "output_dir": output_path,
        "index": index_path,
        "findings": findings_path,
        **feature_paths,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate research report artifacts.")
    parser.add_argument(
        "--feature-report",
        default=str(DEFAULT_FEATURE_REPORT),
        help="Path to feature_selection_report.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where report artifacts will be written.",
    )
    args = parser.parse_args()

    generate_research_report(args.feature_report, args.output_dir)


if __name__ == "__main__":
    main()
