# Project Structure

The current project is organized around separation of concerns:

- `scripts/data/`: data loading and dataset construction.
- `scripts/features/`: feature extraction from text into parquet feature files.
- `scripts/models/`: model definitions and model configuration only.
- `scripts/training/`: training loops and experiment runners.
- `scripts/evaluation/`: shared evaluation helpers and artifact writers.
- `scripts/analysis/`: feature analysis, gate analysis, and report generation.
- `scripts/utils/`: logging and path helpers.

## Top-Level Folders

```text
data/
  original/       raw CSVs
  processed/      cleaned train/val/test CSVs
  features/       extracted semantic, lexical, syntactic, structural, affective features
  lexicons/       external lexicons such as NRC-VAD

artifacts/
  models/         trained model weights, scalers, checkpoints

results/
  analysis/       EDA, feature statistics, feature selection reports
  models/         metrics, summaries, confusion matrices, logs

notebooks/        Colab notebooks for heavy feature/model extraction
scripts/          reusable project code
tests/            unit tests
docs/             research and engineering documentation
```

## Model Output Layout

Model outputs are grouped by paradigm and model name:

```text
results/models/<paradigm>/<model>/
  training/
  evaluation/
  runs/seedN/

artifacts/models/<paradigm>/<model>/
  checkpoints/
  runs/seedN/checkpoints/
```

The intent is simple: `results/` contains readable experiment outputs, while
`artifacts/` contains heavier files needed to reload models.

## Current Paradigms

- `traditional`: Logistic Regression, Random Forest, SVM, XGBoost.
- `lm_based`: MentalRoBERTa fine-tuning and checkpoint storage.
- `fusion`: ConcatMLP, GatedFusion, CrossAttentionFusion.

## Scripts Folder Assessment

The main split is now mostly correct:

- Data-specific objects are in `scripts/data/`.
- Feature extractors are in `scripts/features/`.
- Neural and traditional architectures are in `scripts/models/`.
- Training loops are in `scripts/training/`.
- Metric and report writing logic is in `scripts/evaluation/`.

The remaining rough edges are documented in [07_code_audit.md](07_code_audit.md).

