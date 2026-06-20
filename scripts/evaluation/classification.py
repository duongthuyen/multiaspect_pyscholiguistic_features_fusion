"""Classification evaluation helpers: compute metrics and persist artifacts.

Shared by the training modules; kept here so all evaluation logic lives under
scripts/evaluation/.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import classification_report, f1_score

from scripts import config
from scripts.evaluation.metrics import save_confusion_matrix_artifacts
from scripts.utils.outputs import evaluation_dir

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    input_config: str,
    model_name: str,
    split: str,
) -> dict:
    class_names = [config.ID_TO_CLASS[i] for i in range(config.NUM_LABELS)]
    acc = float((y_pred == y_true).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm_artifacts = save_confusion_matrix_artifacts(
        y_true,
        y_pred,
        class_names,
        evaluation_dir(input_config, model_name) / split / "confusion_matrix",
    )

    return {
        "model_type": model_name,
        "input_config": input_config,
        "split": split,
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 6),
                "recall": round(report[cls]["recall"], 6),
                "f1": round(report[cls]["f1-score"], 6),
                "support": int(report[cls]["support"]),
            }
            for cls in class_names
        },
        "class_names": class_names,
        "confusion_matrix": cm_artifacts,
    }


def save_metrics(result: dict, input_config: str, model_name: str, split: str) -> None:
    eval_root = evaluation_dir(input_config, model_name) / split
    eval_root.mkdir(parents=True, exist_ok=True)

    json_path = eval_root / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    txt_path = eval_root / "summary.txt"
    lines = [
        f"Evaluation - {model_name}  features={input_config}  split={split}",
        "=" * 60,
        f"Accuracy    : {result['accuracy']:.4f}  ({result['accuracy'] * 100:.2f}%)",
        f"Macro F1    : {result['macro_f1']:.4f}",
        f"Weighted F1 : {result['weighted_f1']:.4f}",
        "",
        f"{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}",
        "-" * 52,
    ]
    for cls in result["class_names"]:
        pc = result["per_class"][cls]
        lines.append(
            f"{cls:<12} {pc['precision']:>10.4f} {pc['recall']:>8.4f} "
            f"{pc['f1']:>8.4f} {pc['support']:>9}"
        )
    lines += [
        "",
        f"CM raw CSV : {result['confusion_matrix']['raw_csv_path']}",
        f"CM plot    : {result['confusion_matrix']['plot_path']}",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
