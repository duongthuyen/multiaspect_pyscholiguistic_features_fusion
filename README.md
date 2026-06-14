# Multi-Aspect Psycholinguistic Features for Mental Health Text Classification via Gated Fusion

Graduation thesis — Hanoi University of Science and Technology, 2026.

**Student:** Dương Thị Huyền (MSSV 20224317)  
**Supervisor:** Ph.D. Do Thi Ngoc Diep

---

## Overview

This project classifies Reddit posts into six mental health categories (ADHD, Anxiety, Bipolar, Depression, PTSD, None) by combining a domain-adapted language model with interpretable psycholinguistic features through a gated fusion network.

The pipeline has five stages:

1. **Preprocessing** — merge title + body, clean URLs
2. **Feature extraction** — five complementary feature groups (828 dims total)
3. **Feature analysis** — eta-squared effect size, linguistic profiling, feature selection (60 → 42 features retained)
4. **Fusion models** — two fusion strategies evaluated against a MentalRoBERTa baseline
5. **Evaluation & gate audit** — macro F1, per-class F1, confusion analysis, branch-weight inspection

### Feature Groups

| Group | Dims | What it captures |
|---|---|---|
| Semantic | 768 | MentalRoBERTa CLS embedding (fine-tuned) |
| Lexical | 11 | MTLD, word rates, pronoun rates, punctuation |
| Syntactic | 8 | Dependency complexity, POS ratios, readability |
| Structural | 7 | Sentence coherence, topic drift, tense distribution |
| Affective | 34 | GoEmotions (28) + NRC-VAD (3) + VADER (3) |

### Models

| Model | Description |
|---|---|
| `MentalRoBERTa` | Fine-tuned encoder-only baseline (CLS → Linear) |
| `ConcatMLP` | Naive concatenation of all branches → MLP classifier |
| `GatedFusion` | Three branches (semantic, affective, handcrafted) combined by a class-aware learned gate |

---

## Results

Evaluated across five random seeds (0, 1, 2, 3, 42):

| Model | Accuracy | Macro F1 |
|---|---|---|
| MentalRoBERTa (baseline) | 0.8871 | 0.8873 |
| Concat+MLP | 0.8886 ± 0.0022 | 0.8874 ± 0.0023 |
| Gated Fusion | 0.8870 ± 0.0017 | 0.8858 ± 0.0016 |

All three models are statistically equivalent (within seed-to-seed variance). Gated Fusion's value is interpretability: its gate weights expose per-class branch reliance. The most consistent per-class gains are on PTSD (+0.009) and Anxiety (+0.006).

---

## Requirements

- Python 3.10
- PyTorch 2.4
- CUDA GPU recommended for fine-tuning and feature extraction (CPU sufficient for fusion training)
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

> **NRC-VAD Lexicon** — required for affective VAD features. Download from http://saifmohammad.com/WebPages/nrc-vad.html and place at `data/lexicons/NRC-VAD-Lexicon.txt`.

---

## Pipeline

Run these steps in order from raw data.

### Step 1 — Preprocess

```bash
python -m scripts.data.preprocessing
```

Reads `data/original/both_{train,val,test}.csv`, merges title + post, cleans URLs, writes to `data/processed/`.

### Step 2 — Fine-tune MentalRoBERTa

```bash
python -m scripts.features.semantic.finetune_mental_roberta --epochs 5 --batch 16 --lr 2e-5
```

Saves the fine-tuned backbone to `results/models/roberta/finetuned/`. This backbone is shared by all fusion models.

### Step 3 — Extract features

```bash
python -m scripts.features.extract --input data/processed/train.csv --split train
python -m scripts.features.extract --input data/processed/val.csv   --split val
python -m scripts.features.extract --input data/processed/test.csv  --split test
```

Force re-extraction of a single group:

```bash
python -m scripts.features.extract --input data/processed/train.csv --split train \
    --components affective --force
```

### Step 4 — Train fusion models

**Single seed:**

```bash
python -m scripts.run_one_seed --model gated_fusion --seed 42
python -m scripts.run_one_seed --model concat_mlp   --seed 42
```

**Multi-seed (5 seeds, used for reported results):**

```bash
python -m scripts.run_multi_seed --model gated_fusion
python -m scripts.run_multi_seed --model concat_mlp
```

Key hyperparameters (all defaults in `scripts/config.py`):

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 5e-4 |
| Max epochs | 20 |
| Batch size | 32 |
| Early-stop patience | 2 |
| Label smoothing | 0.1 |
| Projection dim | 256 |
| Gate hidden dim | 128 |
| Handcrafted dropout | 0.4 |
| Diversity penalty λ_div | 0.01 |
| Auxiliary weight λ_aux | 0.3 |

### Step 5 — Evaluate

```bash
python -m scripts.models.fusion.evaluate --model gated_fusion --seed 42 --split test
```

---

## Analysis

```bash
# Gate weight audit — per-class branch routing
python -m scripts.analysis.branch_weights

# Feature statistics — per-class mean heatmaps
python -m scripts.analysis.feature_statistics

# Exploratory data analysis
python -m scripts.data.EDA
```

---

## Output Structure

```
results/
  mental_roberta/
    evaluation/
      finetune/summary.json
      confusion_matrix/
  gated_fusion_seed{N}/         # N = 0, 1, 2, 3, 42
    training/
      checkpoints/best.pt
      checkpoints/handcrafted_scaler.joblib
      logs/history.json
    evaluation/
      summary.json
      gate_weight_range.json
      confusion_matrix/
  concat_mlp_seed{N}/
    training/
    evaluation/
  multi_seed_summary.json       # Aggregated mean ± std across seeds
```

---

## Project Structure

```
scripts/
  config.py                  ← All paths, constants, hyperparameters
  data/
    preprocessing.py
    EDA.py
  features/
    extract.py               ← CLI for batch feature extraction
    orchestrator.py
    semantic/
    lexical/
    syntactic/
    structural/
    affective/
  models/
    fusion/
      blocks.py              ← ProjectionBlock, ClassifierHead
      gated.py               ← GatedFusion and ConcatMLP model definitions
      train.py               ← Training loop
      evaluate.py            ← Evaluation from checkpoint
      feature_loader.py      ← Parquet → tensor loading
  analysis/
    branch_weights.py
    feature_statistics.py
  evaluation/
    metrics.py
  run_one_seed.py
  run_multi_seed.py
tests/
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```
