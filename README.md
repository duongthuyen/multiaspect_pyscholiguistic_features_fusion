# A Multi-Aspect Linguistic Feature Fusion Network for Mental Health Text Classification

Graduation research project at Hanoi University of Science and Technology.

## Overview

This project classifies mental health conditions from Reddit posts using five feature groups:

- **Semantic** - MentalRoBERTa embeddings
- **Lexical** - MTLD, word rates, pronouns, punctuation markers
- **Syntactic** - Complexity, POS ratios, readability
- **Affective** - GoEmotions scores, VAD scores, sentiment arc
- **Structural** - Sentence coherence, topic drift, coherence breaks, tense distribution

The neural fusion models are **Late Concatenation** and **Gated Fusion**. Training can use either a single feature group or the full fused feature set. Classical baselines are also available: Logistic Regression, Random Forest, Support Vector Machine, and XGBoost.

## Dataset

Reddit posts from the Murarka et al. (2021) dataset covering 6 classes:
ADHD, Anxiety, Bipolar, Depression, PTSD, None (control).

## Setup

```bash
python -m venv amh_venv
amh_venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Training

Train a fusion model on all features:

```bash
python -m scripts.models.train_fusion --model concat --features fused
python -m scripts.models.train_fusion --model gated --features fused
```

Train the same fusion pipeline on a single feature group:

```bash
python -m scripts.models.train_fusion --model concat --features semantic
python -m scripts.models.train_fusion --model gated --features lexical
```

Train classical classifiers:

```bash
python -m scripts.models.classical.logistic_regression --features fused
python -m scripts.models.classical.random_forest --features semantic
python -m scripts.models.classical.support_vector_machine --features syntactic
python -m scripts.models.classical.xgboost --features affective
```

Valid `--features` values are `semantic`, `lexical`, `syntactic`, `structural`, `affective`, and `fused`.

## Evaluation

Fusion models can be evaluated separately after training:

```bash
python -m scripts.models.evaluate_fusion --model concat --features fused --split test
python -m scripts.models.evaluate_fusion --model gated --features semantic --split val
```

Each evaluation saves accuracy/F1 metrics plus confusion matrix artifacts as a raw CSV, raw JSON, and PNG plot.

## Output Structure

Outputs are grouped first by input configuration, then by model when needed:

```text
results/
  semantic/
    late_concat/
      training/
        checkpoints/
        logs/
      evaluation/
        metrics.json
        summary.txt
        confusion_matrix/
    logistic_regression/
      training/
      evaluation/
  fused/
    late_concat/
      training/
      evaluation/
    gated/
      training/
      evaluation/
```

Single feature folders such as `lexical/`, `syntactic/`, `structural/`, and `affective/` follow the same pattern: the feature folder contains one subfolder per trained model, and each model has separate `training/` and `evaluation/` directories.

## Project Structure

```text
scripts/
  config.py
  data/
  features/
    semantic/
    lexical/
    syntactic/
    affective/
    structural/
  models/
    fusion/
    classical/
    train_fusion.py
    evaluate_fusion.py
  evaluation/
  analysis/
```
