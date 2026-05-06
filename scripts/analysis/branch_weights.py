"""
Branch importance analysis for trained fusion models.

GatedFusion:
    Runs the test set through the model with return_gates=True and averages
    the softmax gate values across all samples.  Each gate value is the
    fraction of the fused representation contributed by that branch
    (semantic / affective / handcrafted).  Values sum to 1.0 per dimension;
    we report the mean across all 256 hidden dimensions.

LateConcatFusion:
    Has no explicit gate.  Branch importance is approximated from the first
    linear layer of the ClassifierHead (448 → 256).  We slice the weight
    matrix by branch (semantic: cols 0-255, affective: 256-383, handcrafted:
    384-447) and compute mean absolute weight per branch, then normalise to
    sum to 1.0 so the three numbers are directly comparable.

Usage (from project root):
    python -m scripts.analysis.branch_weights --model concat
    python -m scripts.analysis.branch_weights --model gated
    python -m scripts.analysis.branch_weights --model concat --model gated
    python -m scripts.analysis.branch_weights          # runs both

Results are saved to:
    results/fused/{model}/evaluation/branch_weights.json
    results/fused/{model}/evaluation/branch_weights.txt
"""

import argparse
import json
import logging

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts import config
from scripts.models.fusion.factory import build_fusion_model
from scripts.models.fusion.feature_loader import load_fusion_feature_tensors
from scripts.models.train_fusion import load_labels
from scripts.utils.logging_utils import setup_logging
from scripts.utils.outputs import checkpoint_dir, evaluation_dir

logger = logging.getLogger(__name__)

BRANCHES = ["semantic", "affective", "handcrafted"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_test_loader(model_type: str, batch_size: int = 256) -> DataLoader:
    """Load and scale the test split and return a DataLoader that includes labels."""
    semantic, affective, handcrafted, _ = load_fusion_feature_tensors(split="test")
    labels = load_labels("test")

    scaler_path = checkpoint_dir("fused", model_type) / "handcrafted_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            "Train the model first with train_fusion.py"
        )
    scaler = joblib.load(scaler_path)
    hc_scaled = torch.from_numpy(
        scaler.transform(handcrafted.numpy()).astype(np.float32)
    )
    ds = TensorDataset(semantic, affective, hc_scaled, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _load_model(model_type: str) -> torch.nn.Module:
    ckpt_path = checkpoint_dir("fused", model_type) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train the model first with train_fusion.py"
        )
    model = build_fusion_model(model_type)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# GatedFusion — extract gate values
# ---------------------------------------------------------------------------


def analyze_gated(batch_size: int = 256) -> dict:
    """
    Average softmax gate weights across the entire test set.

    Gate tensor shape per batch: (B, 3, 256)
      dim 0 = batch
      dim 1 = branch  (0=semantic, 1=affective, 2=handcrafted)
      dim 2 = hidden dimension

    We average over batch and hidden dims to get one scalar per branch.
    Labels are collected in the same pass to compute per-class gate averages,
    avoiding a second forward pass over the test set.
    """
    logger.info("=" * 60)
    logger.info("GatedFusion — gate weight analysis")
    logger.info("=" * 60)

    model = _load_model("gated")
    loader = _load_test_loader("gated", batch_size=batch_size)

    all_gates: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for semantic, affective, handcrafted, labels in loader:
            _, gates = model(semantic, affective, handcrafted, return_gates=True)
            all_gates.append(gates.cpu().numpy())
            all_labels.append(labels.numpy())

    # (N, 3, 256)
    all_gates_arr = np.concatenate(all_gates, axis=0)
    all_labels_arr = np.concatenate(all_labels)

    # Mean over samples and hidden dimensions → shape (3,)
    mean_per_branch = all_gates_arr.mean(axis=(0, 2))

    logger.info("Overall mean gate (averaged across test set):")
    for name, val in zip(BRANCHES, mean_per_branch):
        logger.info("  %-14s %.4f  (%.1f%%)", name, val, val * 100)
    logger.info("  Gates are softmaxed across branches; total=%.4f", mean_per_branch.sum())

    # Per-class gate breakdown
    per_class: dict[str, dict[str, float]] = {}
    for branch_idx, branch_name in enumerate(BRANCHES):
        per_class[branch_name] = {}
        for cls_id, cls_name in config.ID_TO_CLASS.items():
            mask = all_labels_arr == cls_id
            val = (
                float(all_gates_arr[mask, branch_idx, :].mean())
                if mask.sum() > 0
                else float("nan")
            )
            per_class[branch_name][cls_name] = round(val, 6)

    logger.info("Per-class mean gates:")
    header = f"  {'Branch':<14}" + "".join(
        f"  {cls:<10}" for cls in config.ID_TO_CLASS.values()
    )
    logger.info(header)
    for branch_name in BRANCHES:
        row = f"  {branch_name:<14}" + "".join(
            f"  {per_class[branch_name][cls]:.4f}    "
            for cls in config.ID_TO_CLASS.values()
        )
        logger.info(row)

    return {
        "model_type": "gated",
        "overall": {
            name: round(float(val), 6)
            for name, val in zip(BRANCHES, mean_per_branch)
        },
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# LateConcatFusion — weight-based branch importance
# ---------------------------------------------------------------------------


def analyze_concat() -> dict:
    """
    Approximate branch importance from the classifier head's weight matrix.

    The first linear layer of ClassifierHead is (448 → 256).
    Weight matrix shape: (256, 448).
    Slices by branch:
        semantic:     cols   0 – 255   (256 dims, SEMANTIC_PROJECTION_DIM)
        affective:    cols 256 – 383   (128 dims, AFFECTIVE_PROJECTION_DIM)
        handcrafted:  cols 384 – 447   ( 64 dims, HANDCRAFTED_PROJECTION_DIM)

    Mean absolute weight per branch, then normalised to sum to 1.0.
    """
    logger.info("=" * 60)
    logger.info("LateConcatFusion — classifier weight-based branch importance")
    logger.info("=" * 60)

    model = _load_model("concat")

    # classifier.layers[0] is the first nn.Linear inside ClassifierHead
    W = model.classifier.layers[0].weight.detach().numpy()  # (256, 448)

    s_end = config.SEMANTIC_PROJECTION_DIM                    # 256
    a_end = s_end + config.AFFECTIVE_PROJECTION_DIM           # 384
    h_end = a_end + config.HANDCRAFTED_PROJECTION_DIM         # 448

    slices = {
        "semantic": W[:, :s_end],
        "affective": W[:, s_end:a_end],
        "handcrafted": W[:, a_end:h_end],
    }

    raw = {name: float(np.abs(arr).mean()) for name, arr in slices.items()}
    total = sum(raw.values())
    normalised = {name: round(val / total, 6) for name, val in raw.items()}

    dims = {
        "semantic": s_end,
        "affective": config.AFFECTIVE_PROJECTION_DIM,
        "handcrafted": config.HANDCRAFTED_PROJECTION_DIM,
    }
    logger.info("Normalised branch importance (from classifier head weights):")
    for name in BRANCHES:
        logger.info(
            "  %-14s mean|w|=%.5f  normalised=%.4f  (%d projection dims)",
            name, raw[name], normalised[name], dims[name],
        )
    logger.info("  (approximation — LateConcatFusion has no explicit gate)")

    return {
        "model_type": "concat",
        "method": "mean_abs_weight_of_classifier_head_layer0",
        "raw_mean_abs_weight": raw,
        "normalised": normalised,
        "projection_dims": dims,
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(result: dict, model_type: str) -> None:
    eval_dir = evaluation_dir("fused", model_type)
    eval_dir.mkdir(parents=True, exist_ok=True)

    json_path = eval_dir / "branch_weights.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    txt_path = eval_dir / "branch_weights.txt"
    lines = [f"Branch importance — {model_type.upper()}", "=" * 50]

    if model_type == "gated":
        lines.append("\nOverall mean gate (averaged across test set):")
        for name, val in result["overall"].items():
            lines.append(f"  {name:<14} {val:.4f}  ({val*100:.1f}%)")
        lines.append("\nPer-class mean gate:")
        header = f"  {'Branch':<14}" + "".join(
            f"  {c:<10}" for c in config.ID_TO_CLASS.values()
        )
        lines.append(header)
        for branch in BRANCHES:
            row = f"  {branch:<14}" + "".join(
                f"  {result['per_class'][branch][c]:.4f}    "
                for c in config.ID_TO_CLASS.values()
            )
            lines.append(row)
    else:
        lines.append("\nNormalised branch importance (from classifier head weights):")
        for name, val in result["normalised"].items():
            lines.append(f"  {name:<14} {val:.4f}  ({val*100:.1f}%)")
        lines.append("\n(approximation — no explicit gate in LateConcatFusion)")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved -> %s", json_path)
    logger.info("Saved -> %s", txt_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["concat", "gated"],
        action="append",
        dest="models",
        help="Which model to analyse. Can be passed twice. Defaults to both if omitted.",
    )
    args = parser.parse_args()
    models_to_run = args.models or ["concat", "gated"]

    for model_type in models_to_run:
        try:
            if model_type == "gated":
                result = analyze_gated()
            else:
                result = analyze_concat()
            save_results(result, model_type)
        except FileNotFoundError as exc:
            logger.warning("SKIP %s: %s", model_type, exc)
