"""Multi-seed evaluation for the TRADITIONAL paradigm (mean +/- std).

Mirrors run_multi_seed.py (fusion). Trains each traditional classifier across
config.SEEDS on the TF-IDF (optionally + LSA/SVD) representation and reports
mean +/- std of test accuracy / macro-F1. Per-seed artifacts go to
results/models/traditional/<model>/runs/seed<N>/ ; the aggregate to
results/models/traditional/_summaries/multi_seed_summary.json.

Usage:
    python -m scripts.main run traditional-multi-seed --svd 300
    python -m scripts.main run traditional-multi-seed --models logistic_regression,xgboost
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from scripts import config
from scripts.training.traditional import train_traditional_classifier
from scripts.models.traditional import (
    logistic_regression as lr,
    random_forest as rf,
    support_vector_machine as svm,
    xgboost as xgb,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

BUILDERS = {
    "logistic_regression": lr.build_model,
    "support_vector_machine": svm.build_model,
    "random_forest": rf.build_model,
    "xgboost": xgb.build_model,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd", type=int, default=None,
                    help="TruncatedSVD/LSA components (recommended for rf/svm/xgboost).")
    ap.add_argument("--models", default=",".join(BUILDERS),
                    help="Comma-separated subset of: " + ",".join(BUILDERS))
    ap.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    summary: dict = {"svd_components": args.svd, "seeds": seeds, "models": {}}

    for name in models:
        build = BUILDERS[name]
        per_seed = []
        for seed in seeds:
            est = build(seed)
            run_name = f"{name}/runs/seed{seed}"
            res = train_traditional_classifier(
                est, run_name, svd_components=args.svd, seed=seed
            )
            per_seed.append(res["test"])
            print(f"[{name} seed={seed}] test acc={res['test']['accuracy']:.4f} "
                  f"macroF1={res['test']['macro_f1']:.4f}")
        accs = [p["accuracy"] for p in per_seed]
        f1s = [p["macro_f1"] for p in per_seed]
        summary["models"][name] = {
            "test_accuracy_mean": float(np.mean(accs)),
            "test_accuracy_std": float(np.std(accs)),
            "test_macro_f1_mean": float(np.mean(f1s)),
            "test_macro_f1_std": float(np.std(f1s)),
            "per_seed": per_seed,
        }
        print(f"== {name}: acc {np.mean(accs):.4f}±{np.std(accs):.4f}  "
              f"macroF1 {np.mean(f1s):.4f}±{np.std(f1s):.4f} ==")

    out = config.RESULTS_DIR / "models" / "traditional" / "_summaries" / "multi_seed_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved", out)


if __name__ == "__main__":
    main()
