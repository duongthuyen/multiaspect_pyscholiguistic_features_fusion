# Code Audit

This audit focuses on organization and maintainability. It does not change
reported results.

## What Is Organized Well

- `scripts/data/` now owns processed CSV loading, parquet loading, and dataset
  construction.
- `scripts/features/` owns feature extraction and feature combination.
- `scripts/models/` contains architecture definitions and model builders.
- `scripts/training/` contains training loops and experiment runners.
- `scripts/evaluation/` contains shared metric and artifact writing helpers.
- `results/` and `artifacts/` are now separated: metrics/logs vs checkpoints.

## Current Pipeline Coverage

| Paradigm | Data Loader | Model Definition | Training | Evaluation | Artifacts |
|---|---|---|---|---|---|
| Traditional | yes | yes | yes | yes | yes |
| MentalRoBERTa | yes | yes | yes | yes | yes |
| ConcatMLP | yes | yes | yes | yes | yes |
| GatedFusion | yes | yes | yes | yes | yes |
| CrossAttentionFusion | yes | yes | yes | aggregate summary | yes |

## Known Rough Edges

1. Some tests still reflect an older fusion API.
   - `tests/test_fusion_models.py` mentions old names such as
     `ContentGatedFusion`, `LoadBalancedFusion`, `content_gate`, and
     `load_balance`.
   - `tests/models/test_evaluate_fusion.py` expects the old
     `results/gated_fusion/<variant>/...` layout.

2. `scripts/analysis/compare_runs.py` still describes an older selected-feature
   comparison workflow. It should be revisited if selected-feature training is
   still part of the thesis pipeline.

3. CrossAttentionFusion is implemented as a runner rather than a standard
   train/evaluate pair. It works and now saves artifacts, but its interface is
   less uniform than GatedFusion and ConcatMLP.

4. Historical JSON files may contain old absolute paths inside their content.
   The files are in the new location, but old internal path strings can remain
   in summaries generated before the reorganization.

5. `PROJECT_STRUCTURE.md` appears to contain mojibake/encoding artifacts.
   The new `docs/` folder should be treated as the canonical documentation.

## Non-Result-Changing Cleanup Recommendations

- Update stale tests to the current model names and output layout.
- Standardize CrossAttentionFusion into `scripts/training/cross_attention_train.py`
  and optionally `scripts/evaluation/cross_attention_evaluate.py`.
- Add a small result-index generator that scans `results/models/**/summary.json`
  and writes one overview table.
- Keep `feature_loader.py` numpy-only; tensor conversion should remain in
  `fusion_dataset.py`.
- Avoid moving checkpoints back into `results/`.

