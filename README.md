# Mental Health Fusion

Multi-aspect mental health text classification with MentalRoBERTa,
psycholinguistic features, and fusion models.

The project classifies Reddit posts into six labels:

```text
ADHD, Anxiety, Bipolar, Depression, PTSD, None
```

It compares three modeling paradigms:

- `traditional`: TF-IDF + dense psycholinguistic features.
- `lm_based`: fine-tuned MentalRoBERTa.
- `fusion`: semantic, affective, and handcrafted feature fusion.

## Documentation

The detailed research and engineering notes are split across `docs/`:

- [Project structure](docs/01_project_structure.md)
- [Research background](docs/02_research_background.md)
- [Data and features](docs/03_data_and_features.md)
- [Models](docs/04_models.md)
- [Training and evaluation](docs/05_training_evaluation.md)
- [Results summary](docs/06_results_summary.md)
- [Code audit](docs/07_code_audit.md)
- [Reproducibility](docs/08_reproducibility.md)

Start from [docs/INDEX.md](docs/INDEX.md) for the full map.

## Current Layout

```text
data/
  original/       raw inputs
  processed/      cleaned train/val/test CSVs
  features/       extracted feature parquet files
  lexicons/       NRC-VAD and other lexicons

scripts/
  data/           loaders and dataset builders
  features/       feature extraction
  models/         model definitions
  training/       training loops and runners
  evaluation/     metrics and evaluation artifacts
  analysis/       EDA, feature analysis, report generation
  utils/          logging and path helpers

results/
  analysis/       plots, feature statistics, reports
  models/         metrics, summaries, confusion matrices, logs

artifacts/
  models/         checkpoints, scalers, saved model files
```

## Quick Setup

```powershell
python -m venv amh_venv
.\amh_venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

NRC-VAD is required for VAD features:

```text
data/lexicons/NRC-VAD-Lexicon.txt
```

## Main Commands

Use `scripts.main` as the primary entry point.

```powershell
python -m scripts.main preprocess

python -m scripts.main extract --input data/processed/train.csv --split train
python -m scripts.main extract --input data/processed/val.csv --split val
python -m scripts.main extract --input data/processed/test.csv --split test

python -m scripts.main combine
```

Train traditional baselines:

```powershell
python -m scripts.main train classical --model logistic_regression --features traditional --seed 42
python -m scripts.main train classical --model random_forest --features traditional --svd 300 --seed 42
python -m scripts.main train classical --model support_vector_machine --features traditional --svd 300 --seed 42
python -m scripts.main train classical --model xgboost --features traditional --svd 300 --seed 42
```

Train fusion models:

```powershell
python -m scripts.main train fusion
python -m scripts.training.concat_train
python -m scripts.main run cross-attention
```

Fine-tune MentalRoBERTa:

```powershell
python -m scripts.features.semantic.finetune_mental_roberta --epochs 5 --batch 16 --lr 2e-5
```

Evaluate and analyze:

```powershell
python -m scripts.main evaluate --split test
python -m scripts.main analyze-features --select
python -m scripts.main analyze-branches
python -m scripts.main report-pack
```

## Output Paths

Results:

```text
results/models/<paradigm>/<model>/evaluation/
results/models/<paradigm>/<model>/training/
results/models/<paradigm>/<model>/runs/seedN/
```

Artifacts:

```text
artifacts/models/<paradigm>/<model>/checkpoints/
artifacts/models/<paradigm>/<model>/runs/seedN/checkpoints/
```

## Current Result Snapshot

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| MentalRoBERTa | 0.8871 | 0.8873 |
| Logistic Regression | 0.8044 | 0.8041 |
| Random Forest | 0.7211 | 0.7183 |
| Support Vector Machine | 0.8145 | 0.8146 |
| XGBoost | 0.7997 | 0.8002 |
| ConcatMLP | 0.8923 | 0.8911 |
| GatedFusion | 0.8891 | 0.8892 |
| CrossAttentionFusion | 0.8860 +/- 0.0020 | 0.8861 +/- 0.0019 |

See [docs/06_results_summary.md](docs/06_results_summary.md) for details.

## Tests

Core smoke tests:

```powershell
.\amh_venv\Scripts\python.exe -m pytest tests\data\test_feature_loader.py tests\features\semantic\test_finetune_mental_roberta.py tests\test_metrics.py tests\test_outputs.py -q
```

Some older tests still reference pre-refactor fusion APIs and output paths.
See [docs/07_code_audit.md](docs/07_code_audit.md).

