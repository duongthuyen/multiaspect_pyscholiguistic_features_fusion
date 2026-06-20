# Results Summary

This file records the current result snapshot in the workspace. Re-running
training may change values slightly due to seed and hardware details.

## LM-Based Baseline

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| MentalRoBERTa | 0.8871 | 0.8873 |

Path:

```text
results/models/lm_based/mental_roberta/evaluation/summary.json
```

## Traditional Models

All traditional models use `--features traditional`, which means TF-IDF text
features plus 60 dense psycholinguistic features.

| Model | Accuracy | Macro-F1 | Notes |
|---|---:|---:|---|
| Logistic Regression | 0.8044 | 0.8041 | full sparse TF-IDF |
| Random Forest | 0.7211 | 0.7183 | SVD 300 |
| Support Vector Machine | 0.8145 | 0.8146 | SVD 300 |
| XGBoost | 0.7997 | 0.8002 | SVD 300 |

Paths:

```text
results/models/traditional/<model>/evaluation/test/metrics.json
artifacts/models/traditional/<model>/checkpoints/model.joblib
```

## Fusion Models

| Model | Accuracy | Macro-F1 | Run Type |
|---|---:|---:|---|
| ConcatMLP | 0.8923 | 0.8911 | single default run, seed 42 |
| GatedFusion | 0.8891 | 0.8892 | evaluated from checkpoint |
| CrossAttentionFusion | 0.8860 +/- 0.0020 | 0.8861 +/- 0.0019 | 5 seeds |

Paths:

```text
results/models/fusion/concat_mlp/evaluation/summary.json
results/models/fusion/gated_fusion/evaluation/test/metrics.json
results/models/fusion/cross_attention/summary.json
```

## How To Interpret `evaluation/` vs `runs/seedN/`

The top-level `evaluation/` folder is the main/default run for the model,
usually seed 42.

The `runs/seedN/` folders contain per-seed runs from multi-seed experiments.

The `_summaries/` folder stores aggregate files across models or seeds.

