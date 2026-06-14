"""
Compare baseline vs selected-feature training runs for a gated fusion variant.

Reads summary.json from both output directories and prints a side-by-side
comparison table.  Optionally exports the comparison as CSV.

Output directories expected:
    results/gated_fusion/<variant>/evaluation/summary.json          ← baseline
    results/gated_fusion_selected/<variant>/evaluation/summary.json ← selected

Usage
-----
    python -m scripts.main compare --variant content_gate
    python -m scripts.main compare --variant content_gate --format csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from scripts import config

logger = logging.getLogger(__name__)

# Metrics to include in the comparison table (key in summary.json → display label)
_SCALAR_METRICS: list[tuple[str, str]] = [
    ("test_acc",    "Test accuracy"),
    ("macro_f1",    "Test macro F1"),
    ("weighted_f1", "Test weighted F1"),
    ("best_val_acc","Best val accuracy"),
    ("epochs_trained", "Epochs trained"),
    ("best_epoch",  "Best epoch"),
]

_CLASS_NAMES = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]


def _load_summary(variant_dir: Path) -> dict | None:
    """Load evaluation/summary.json from a variant output directory."""
    path = variant_dir / "evaluation" / "summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _delta_str(baseline: float | None, selected: float | None) -> str:
    """Format the delta between two metric values with a +/- sign."""
    if baseline is None or selected is None:
        return "—"
    delta = selected - baseline
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:+.4f}"


def compare_runs(variant: str, output_format: str = "table") -> dict | None:
    """
    Load both summary files, compute deltas, and print a comparison.

    Parameters
    ----------
    variant : str
        Gated fusion variant name (e.g. "content_gate").
    output_format : {"table", "csv"}
        How to display results.

    Returns
    -------
    dict with keys "baseline", "selected", "delta" — each a dict of metrics.
    None if either run is missing.
    """
    base_dir     = config.RESULTS_DIR / config.GATED_FUSION_OUTPUT_DIR / variant
    selected_dir = config.RESULTS_DIR / (str(config.GATED_FUSION_OUTPUT_DIR) + "_selected") / variant

    baseline = _load_summary(base_dir)
    selected = _load_summary(selected_dir)

    if baseline is None:
        logger.error(
            "Baseline summary not found: %s\n"
            "Train the baseline first:  python -m scripts.main train fusion --variant %s",
            base_dir / "evaluation" / "summary.json", variant,
        )
        return None

    if selected is None:
        logger.error(
            "Selected-feature summary not found: %s\n"
            "Train with selection first:\n"
            "  python -m scripts.main train fusion --variant %s "
            "--select-report results/plots/feature_statistics/feature_selection_report.csv",
            selected_dir / "evaluation" / "summary.json", variant,
        )
        return None

    # ── Build comparison rows ─────────────────────────────────────────────────
    rows: list[dict] = []

    for key, label in _SCALAR_METRICS:
        b_val = baseline.get(key)
        s_val = selected.get(key)
        rows.append({
            "metric":   label,
            "baseline": f"{b_val:.4f}" if isinstance(b_val, float) else str(b_val),
            "selected": f"{s_val:.4f}" if isinstance(s_val, float) else str(s_val),
            "delta":    _delta_str(
                b_val if isinstance(b_val, float) else None,
                s_val if isinstance(s_val, float) else None,
            ),
        })

    # Per-class F1
    b_f1 = baseline.get("per_class_f1", {})
    s_f1 = selected.get("per_class_f1", {})
    for cls in _CLASS_NAMES:
        rows.append({
            "metric":   f"F1 [{cls}]",
            "baseline": f"{b_f1.get(cls, 0.0):.4f}",
            "selected": f"{s_f1.get(cls, 0.0):.4f}",
            "delta":    _delta_str(b_f1.get(cls), s_f1.get(cls)),
        })

    # Feature selection info
    sel_info = selected.get("feature_selection", {})
    rows.append({
        "metric":   "Features dropped",
        "baseline": "0",
        "selected": str(sel_info.get("n_dropped", "?")),
        "delta":    "—",
    })
    rows.append({
        "metric":   "Features kept",
        "baseline": "60",   # total interpretable features (affective + handcrafted)
        "selected": str(sel_info.get("n_kept", "?")),
        "delta":    "—",
    })
    rows.append({
        "metric":   "Mask threshold",
        "baseline": "—",
        "selected": str(sel_info.get("threshold") or "—"),
        "delta":    "—",
    })

    # ── Display ───────────────────────────────────────────────────────────────
    if output_format == "csv":
        import csv, sys
        writer = csv.DictWriter(sys.stdout, fieldnames=["metric", "baseline", "selected", "delta"])
        writer.writeheader()
        writer.writerows(rows)
    else:
        _print_table(variant, rows, sel_info)

    # ── Save comparison CSV to results dir ────────────────────────────────────
    save_dir = config.RESULTS_DIR / config.GATED_FUSION_OUTPUT_DIR / variant / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)
    import csv
    csv_path = save_dir / "comparison_selected_vs_baseline.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "baseline", "selected", "delta"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Comparison saved → %s", csv_path)

    return {
        "baseline": {r["metric"]: r["baseline"] for r in rows},
        "selected": {r["metric"]: r["selected"] for r in rows},
        "delta":    {r["metric"]: r["delta"]    for r in rows},
    }


def _print_table(variant: str, rows: list[dict], sel_info: dict) -> None:
    """Print a formatted comparison table to stdout."""
    col_w = [max(len(r[k]) for r in rows) for k in ("metric", "baseline", "selected", "delta")]
    col_w[0] = max(col_w[0], 28)
    col_w[1] = max(col_w[1], 10)
    col_w[2] = max(col_w[2], 10)
    col_w[3] = max(col_w[3], 8)

    sep = "  ".join("-" * w for w in col_w)
    hdr = "  ".join(h.ljust(col_w[i]) for i, h in enumerate(
        ["Metric", "Baseline", "Selected", "Delta"]
    ))

    dropped = sel_info.get("dropped_names") or []
    print()
    print(f"=== Feature Selection Comparison — {variant} ===")
    if dropped:
        print(f"    Dropped features ({len(dropped)}): {', '.join(dropped)}")
    print()
    print(hdr)
    print(sep)
    for row in rows:
        delta = row["delta"]
        # Colour-code delta: positive F1 gains are highlighted
        if delta.startswith("+") and float(delta) > 0:
            delta_display = f"▲ {delta}"
        elif delta.startswith("-"):
            delta_display = f"▼ {delta}"
        else:
            delta_display = f"  {delta}"
        print("  ".join([
            row["metric"].ljust(col_w[0]),
            row["baseline"].ljust(col_w[1]),
            row["selected"].ljust(col_w[2]),
            delta_display.ljust(col_w[3] + 2),
        ]))
    print()
    print("Legend:  ▲ selected > baseline   ▼ selected < baseline")
    print("         Positive delta on F1 metrics = feature selection helped.")
    print()
