# A Multi-Aspect Linguistic Feature Fusion Network for Mental Health Text Classification

Graduation research project — Hanoi University of Science and Technology.

## Overview

This project classifies mental health conditions from Reddit posts (ADHD, Anxiety, Bipolar, Depression, PTSD, None) by extracting five complementary feature groups and combining them with a learnable gated fusion network.

| Feature group | Dims | What it captures |
|---|---|---|
| **Semantic** | 768 | MentalRoBERTa CLS embeddings |
| **Lexical** | 11 | MTLD, word rates, pronouns, punctuation |
| **Syntactic** | 8 | Dependency complexity, POS ratios, readability |
| **Affective** | 34 | GoEmotions scores, VAD scores, VADER sentiment |
| **Structural** | 7 | Sentence coherence, topic drift, tense distribution |

Four gated fusion variants are implemented on top of these features:

| Variant | Description |
|---|---|
| `content_gate` | Gate MLP sees all three raw feature branches simultaneously |
| `class_aware` | Gate also receives soft class probabilities from an auxiliary semantic head |
| `load_balance` | Extends `class_aware` with a diversity penalty that discourages gate uniformity |
| `per_class_gate` | Each class has its own set of branch weights, mixed by auxiliary class probabilities |

Classical baselines (Logistic Regression, Random Forest, SVM, XGBoost) are available separately.

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- See `requirements.txt` for the full dependency list

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv amh_venv
amh_venv\Scripts\activate        # Windows
# source amh_venv/bin/activate   # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy language model
python -m spacy download en_core_web_sm
```

> **NRC-VAD Lexicon** — the affective VAD extractor requires the NRC Valence-Arousal-Dominance lexicon. Download it from http://saifmohammad.com/WebPages/nrc-vad.html and place it at `data/lexicons/NRC-VAD-Lexicon.txt`.

## Full Pipeline

Run these steps in order when starting from raw data.

### Step 1 — Preprocess raw CSV files

```bash
python -m scripts.data.preprocessing
```

Reads `data/original/both_{train,val,test}.csv`, merges title + post, cleans URLs and whitespace, writes to `data/processed/{train,val,test}.csv`.

### Step 2 — (Optional) Fine-tune MentalRoBERTa backbone

Skip this step if you want to use the pre-trained `mental/mental-roberta-base` weights directly.

```bash
python -m scripts.features.semantic.finetune_mental_roberta
# With overrides:
python -m scripts.features.semantic.finetune_mental_roberta --epochs 5 --batch 16 --lr 2e-5
```

Saves the fine-tuned backbone to `results/models/roberta/finetuned/`.

### Step 3 — Extract features

```bash
# Extract all groups for all splits (run once per split)
python -m scripts.features.extract --input data/processed/train.csv --split train
python -m scripts.features.extract --input data/processed/val.csv   --split val
python -m scripts.features.extract --input data/processed/test.csv  --split test
```

Partial extraction or forced re-extraction:

```bash
# Re-extract only the affective group
python -m scripts.features.extract --input data/processed/train.csv --split train \
    --components affective --force

# Extract only VAD and VADER sub-features
python -m scripts.features.extract --input data/processed/train.csv --split train \
    --components "affective.vad,affective.vader" --force
```

### Step 4 — Train gated fusion models

```bash
python -m scripts.models.fusion.train --variant content_gate
python -m scripts.models.fusion.train --variant class_aware
python -m scripts.models.fusion.train --variant load_balance
python -m scripts.models.fusion.train --variant per_class_gate
```

Override any hyperparameter from the command line:

```bash
python -m scripts.models.fusion.train --variant load_balance --epochs 30 --lr 1e-3
```

All defaults are defined in `scripts/config.py`. Full list of CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--variant` | `content_gate` | One of the four supported variants |
| `--features` | `fused` | Feature subset: `semantic`, `lexical`, `syntactic`, `structural`, `affective`, `fused` |
| `--epochs` | 20 | Maximum training epochs |
| `--lr` | 5e-4 | Adam learning rate |
| `--batch` | 32 | Batch size |
| `--seed` | 42 | Random seed |
| `--label_smoothing` | 0.1 | CrossEntropyLoss label smoothing |
| `--gate_weight_decay` | 1e-4 | Weight decay for gate parameters |
| `--early_stopping_patience` | 2 | Stop if val loss does not improve for N epochs |
| `--aux_weight` | 0.3 | Auxiliary CE loss coefficient (class_aware, load_balance, per_class_gate) |
| `--projection_dim` | 256 | Branch projection output dimension |
| `--gate_hidden_dim` | 128 | Gate MLP hidden dimension |
| `--handcrafted_dropout` | 0.4 | Dropout on the handcrafted branch |

### Step 5 — Evaluate a trained model

```bash
python -m scripts.models.fusion.evaluate --variant content_gate --split test
python -m scripts.models.fusion.evaluate --variant class_aware  --split val
```

### Step 6 — Classical baselines

```bash
python -m scripts.models.classical.logistic_regression    --features semantic
python -m scripts.models.classical.random_forest          --features semantic
python -m scripts.models.classical.support_vector_machine --features semantic
python -m scripts.models.classical.xgboost                --features semantic
```

The `--features` flag accepts: `semantic`, `lexical`, `syntactic`, `structural`, `affective`, `fused`.

## Analysis Scripts

```bash
# Gate weight analysis — per-class branch routing from a saved checkpoint
python -m scripts.analysis.branch_weights --variant content_gate

# Feature statistics — scaled heatmaps (standardised features)
python -m scripts.analysis.feature_statistics

# Feature statistics — raw violin plots per sub-extractor
python -m scripts.analysis.feature_statistics --raw

# Exploratory data analysis (class distribution, word count KDE)
python -m scripts.data.analysis
```

## Output Structure

```
results/
  gated_fusion/
    {variant}/                    # content_gate | class_aware | load_balance | per_class_gate
      training/
        checkpoints/
          best.pt                 # Model weights at best val_acc epoch
          handcrafted_scaler.joblib
        logs/
          train.log
          history.json
          gate_stats_per_epoch.jsonl
        training_history.csv
      evaluation/
        summary.json              # Full metrics + gate stats
        summary.txt               # Human-readable summary
        gate_weights_per_class.csv
        gate_weight_range.json
        classification_report.csv
        confusion_matrix/
          confusion_matrix.csv
          confusion_matrix.json
          confusion_matrix.png
  {feature_config}/               # semantic | fused | ...
    {model_name}/                 # logistic_regression | random_forest | ...
      training/
        checkpoints/model.joblib
        logs/train.log
      evaluation/
        val/metrics.json
        test/metrics.json
  plots/
    feature_statistics/           # Scaled heatmaps per group
    feature_statistics_raw/       # Raw violin plots per sub-extractor
```

## Project Structure

```
scripts/
  config.py                  ← Central config: all paths, constants, hyperparameters
  data/
    preprocessing.py
    analysis.py
  features/
    base.py                  ← FeatureExtractorBase ABC
    extract.py               ← CLI entry point for batch extraction
    orchestrator.py
    combination.py           ← Utility: combine sub-extractor parquets (not in main pipeline)
    semantic/
    lexical/
    syntactic/
    structural/
    affective/
  models/
    fusion/
      blocks.py              ← ProjectionBlock, ClassifierHead
      gated.py               ← Four gated fusion variants + factory
      feature_loader.py      ← Parquet → tensor loading
      train.py               ← Training loop
      evaluate.py            ← Standalone evaluation from checkpoint
    classical/
      common.py
      logistic_regression.py
      random_forest.py
      support_vector_machine.py
      xgboost.py
  evaluation/
    metrics.py               ← Confusion matrix artifacts
  analysis/
    branch_weights.py
    feature_statistics.py
  utils/
    logging_utils.py
    outputs.py               ← Path helpers for classical model results
    combined_features_info.py  ← Utility: inspect combined.parquet files
tests/
  models/
    test_evaluate_fusion.py
    test_train_fusion.py
  test_fusion_models.py
  test_outputs.py
```

## Benchmark Results

Best classical baseline: **SVM on semantic features** — Macro F1 = **0.8860**

| Model | Features | Accuracy | Macro F1 |
|---|---|---|---|
| Logistic Regression | semantic | 87.50% | 0.8752 |
| Random Forest | semantic | 88.58% | 0.8859 |
| **SVM (RBF)** | **semantic** | **88.58%** | **0.8860** |
| XGBoost | semantic | 88.10% | 0.8812 |

Hardest class: **Bipolar** (most confused with Depression and Anxiety). Easiest class: **None** (F1 ≈ 0.986).

## Running Tests

```bash
python -m pytest tests/ -v
```
