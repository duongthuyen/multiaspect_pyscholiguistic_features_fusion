"""
Multi-seed evaluation: GatedFusion and ConcatMLP.

Trains both models across SEEDS, saves per-seed results under
    results/models/fusion/gated_fusion/runs/seed{N}/
    results/models/fusion/concat_mlp/runs/seed{N}/
then aggregates under
    results/models/fusion/_summaries/

Usage:
    python -m scripts.main run multi-seed
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from scripts import config

logging.basicConfig(
    level=logging.WARNING,          # suppress per-epoch noise
    format="%(levelname)s %(name)s: %(message)s",
)

SEEDS = [42, 0, 1, 2, 3]
CLASS_NAMES = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agg(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 4),
        "std":  round(float(arr.std()),  4),
        "min":  round(float(arr.min()),  4),
        "max":  round(float(arr.max()),  4),
        "values": [round(v, 6) for v in values],
    }


def _per_class_agg(per_seed: list[dict]) -> dict:
    """Aggregate per-class F1 across seeds."""
    out = {}
    for cls in CLASS_NAMES:
        vals = [r["per_class_f1"][cls] for r in per_seed]
        out[cls] = _agg(vals)
    return out


# ---------------------------------------------------------------------------
# GatedFusion run
# ---------------------------------------------------------------------------

def run_gated(seed: int) -> dict:
    # Route output to a seed-specific directory
    out_dir = f"models/fusion/gated_fusion/runs/seed{seed}"
    config.GATED_FUSION_OUTPUT_DIR = out_dir

    # Patch the _output_root in train module so paths resolve correctly
    import scripts.training.fusion_train as train_mod
    train_mod._output_root = lambda cfg=None: config.RESULTS_DIR / out_dir

    cfg = config.get_gated_fusion_config(overrides={"seed": seed})

    print(f"  [GatedFusion seed={seed}] training → results/{out_dir}/")
    result = train_mod.train(cfg)
    print(f"  [GatedFusion seed={seed}] macro_f1={result['macro_f1']:.4f}  acc={result['test_acc']:.4f}")

    return {
        "seed": seed,
        "macro_f1": result["macro_f1"],
        "test_acc": result["test_acc"],
        "per_class_f1": result["per_class_f1"],
        "best_epoch": result["best_epoch"],
        "epochs_trained": result["epochs_trained"],
    }


# ---------------------------------------------------------------------------
# ConcatMLP run
# ---------------------------------------------------------------------------

def run_concat(seed: int) -> dict:
    import scripts.training.concat_train as cb

    out_dir = config.RESULTS_DIR / "models" / "fusion" / "concat_mlp" / "runs" / f"seed{seed}"
    cb.OUTPUT_DIR = out_dir
    config.SEED = seed  # concat reads config.SEED inside train()

    print(f"  [ConcatMLP   seed={seed}] training → results/fusion/concat_mlp_seed{seed}/")
    result = cb.train()
    print(f"  [ConcatMLP   seed={seed}] macro_f1={result['macro_f1']:.4f}  acc={result['test_acc']:.4f}")

    return {
        "seed": seed,
        "macro_f1": result["macro_f1"],
        "test_acc": result["test_acc"],
        "per_class_f1": result["per_class_f1"],
        "best_epoch": result["best_epoch"],
        "epochs_trained": result["epochs_trained"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    gated_per_seed: list[dict] = []
    concat_per_seed: list[dict] = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})")
        print(f"{'='*60}")

        gated_per_seed.append(run_gated(seed))
        concat_per_seed.append(run_concat(seed))

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    summary = {
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "mental_roberta_baseline": {
            "macro_f1": 0.8873,
            "test_acc": 0.8871,
            "note": "Single fine-tuning run on GPU; not re-run across seeds",
        },
        "concat_mlp": {
            **_agg([r["macro_f1"] for r in concat_per_seed]),
            "label": "macro_f1",
            "test_acc": _agg([r["test_acc"] for r in concat_per_seed]),
            "per_class_f1": _per_class_agg(concat_per_seed),
            "per_seed": concat_per_seed,
        },
        "gated_fusion": {
            **_agg([r["macro_f1"] for r in gated_per_seed]),
            "label": "macro_f1",
            "test_acc": _agg([r["test_acc"] for r in gated_per_seed]),
            "per_class_f1": _per_class_agg(gated_per_seed),
            "per_seed": gated_per_seed,
        },
    }

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("MULTI-SEED AGGREGATED RESULTS")
    print(f"Seeds: {SEEDS}")
    print("="*60)

    rob = summary["mental_roberta_baseline"]
    print(f"\nMentalRoBERTa  (single run): macro F1 = {rob['macro_f1']:.4f}")

    for label, key in [("Concat+MLP", "concat_mlp"), ("Gated Fusion", "gated_fusion")]:
        r = summary[key]
        print(f"\n{label} ({len(SEEDS)} seeds):")
        print(f"  Macro F1 : {r['mean']:.4f} ± {r['std']:.4f}  (min {r['min']:.4f}, max {r['max']:.4f})")
        print(f"  Accuracy : {r['test_acc']['mean']:.4f} ± {r['test_acc']['std']:.4f}")
        print(f"  Per-seed F1 : {r['values']}")
        print(f"  Per-class F1 (mean ± std):")
        for cls, ag in r["per_class_f1"].items():
            print(f"    {cls:<12} {ag['mean']:.4f} ± {ag['std']:.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = config.RESULTS_DIR / "models" / "fusion" / "_summaries" / "multi_seed_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {out_path}")


if __name__ == "__main__":
    main()
