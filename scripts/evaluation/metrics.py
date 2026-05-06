from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def save_confusion_matrix_artifacts(
    y_true,
    y_pred,
    class_names: list[str],
    output_dir: Path,
    prefix: str = "confusion_matrix",
) -> dict[str, str | list[list[int]]]:
    """Save confusion matrix as PNG, CSV, and JSON-friendly raw values."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    csv_path = output_dir / f"{prefix}.csv"
    np.savetxt(csv_path, cm, fmt="%d", delimiter=",")

    json_path = output_dir / f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(
            {"class_names": class_names, "confusion_matrix": cm.tolist()},
            f,
            indent=2,
        )

    png_path = output_dir / f"{prefix}.png"
    fig_width = max(7, len(class_names) * 1.1)
    fig_height = max(5, len(class_names) * 0.9)
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    logger.info("Confusion matrix saved -> %s", output_dir)

    return {
        "confusion_matrix": cm.tolist(),
        "raw_csv_path": str(csv_path),
        "raw_json_path": str(json_path),
        "plot_path": str(png_path),
    }
