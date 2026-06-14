"""Branch gate analysis for trained gated fusion variants."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts import config
from scripts.models.fusion.feature_loader import load_feature_tensors
from scripts.models.fusion.gated import build_gated_model
from scripts.models.fusion.train import load_labels
from scripts.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

BRANCHES = ["semantic", "affective", "handcrafted"]


def _variant_root(variant: str | None = None) -> Path:
    return config.RESULTS_DIR / config.GATED_FUSION_OUTPUT_DIR


def _load_test_loader(variant: str, batch_size: int = 256) -> DataLoader:
    semantic, affective, handcrafted, _ = load_feature_tensors(input_config="fused", split="test")
    labels = load_labels("test")

    scaler_path = _variant_root(variant) / "training" / "checkpoints" / "handcrafted_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    hc_scaled = torch.from_numpy(
        scaler.transform(handcrafted.numpy()).astype(np.float32)
    )
    ds = TensorDataset(semantic, affective, hc_scaled, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _load_model(model_cfg: dict) -> torch.nn.Module:
    variant = model_cfg["model"]
    ckpt_path = _variant_root(variant) / "training" / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = build_gated_model(variant, model_cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def analyze_variant(model_cfg: dict, batch_size: int = 256) -> dict:
    variant = model_cfg["model"]
    logger.info("=" * 60)
    logger.info("Gated fusion gate analysis - %s", variant)
    logger.info("=" * 60)

    model = _load_model(model_cfg)
    loader = _load_test_loader(variant, batch_size=batch_size)

    all_gates: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for semantic, affective, handcrafted, labels in loader:
            _, gates = model(semantic, affective, handcrafted, return_gates=True)
            all_gates.append(gates.cpu().numpy())
            all_labels.append(labels.numpy())

    gates_arr = np.concatenate(all_gates, axis=0)
    labels_arr = np.concatenate(all_labels)
    mean_per_branch = gates_arr.mean(axis=0)

    per_class: dict[str, dict[str, float]] = {branch: {} for branch in BRANCHES}
    for branch_idx, branch_name in enumerate(BRANCHES):
        for cls_id, cls_name in config.ID_TO_CLASS.items():
            mask = labels_arr == cls_id
            val = float(gates_arr[mask, branch_idx].mean()) if mask.sum() else float("nan")
            per_class[branch_name][cls_name] = round(val, 6)

    logger.info("Overall mean gate:")
    for name, val in zip(BRANCHES, mean_per_branch):
        logger.info("  %-14s %.4f  (%.1f%%)", name, val, val * 100)

    return {
        "model_name": variant,
        "overall": {
            name: round(float(val), 6)
            for name, val in zip(BRANCHES, mean_per_branch)
        },
        "per_class": per_class,
    }


def save_results(result: dict) -> None:
    eval_dir = _variant_root(result["model_name"]) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    json_path = eval_dir / "branch_weights.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    txt_path = eval_dir / "branch_weights.txt"
    lines = [f"Branch importance - {result['model_name']}", "=" * 50]
    lines.append("\nOverall mean gate:")
    for name, val in result["overall"].items():
        lines.append(f"  {name:<14} {val:.4f}  ({val * 100:.1f}%)")
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
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Saved -> %s", json_path)
    logger.info("Saved -> %s", txt_path)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="gated_fusion")
    args = parser.parse_args()

    model_cfg = config.get_gated_fusion_config(args.variant)
    result = analyze_variant(model_cfg)
    save_results(result)
