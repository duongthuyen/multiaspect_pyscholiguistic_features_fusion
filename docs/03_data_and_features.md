# Data And Features

## Input Data

Raw inputs live in `data/original/` and processed inputs live in
`data/processed/`.

Expected processed files:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

Required processed columns:

- `text`: merged and cleaned post text.
- `class_id`: integer class label.
- `class_name`: human-readable class label.

## Feature Groups

The full fused representation contains five feature groups.

| Group | Dim | Source |
|---|---:|---|
| Semantic | 768 | Fine-tuned MentalRoBERTa CLS embedding |
| Lexical | 11 | diversity, word rates, pronouns, punctuation |
| Syntactic | 8 | complexity, POS ratios, readability |
| Structural | 7 | coherence, tense |
| Affective | 34 | GoEmotions, NRC-VAD, VADER |

Total fused dimension: `828`.

Handcrafted dimension used by fusion models:

```text
lexical 11 + syntactic 8 + structural 7 = 26
```

Traditional dense dimension:

```text
lexical 11 + syntactic 8 + structural 7 + affective 34 = 60
```

## Feature Storage

Feature parquet files live under:

```text
data/features/<group>/<split>/<subfeature>.parquet
```

Examples:

```text
data/features/semantic/train/mental_roberta.parquet
data/features/affective/test/goemotions.parquet
data/features/lexical/val/pronouns.parquet
```

## Loader Responsibilities

- `scripts/data/feature_loader.py`: numpy-only parquet loading.
- `scripts/data/fusion_dataset.py`: tensors for fusion models.
- `scripts/data/traditional_dataset.py`: text + dense frame for traditional models.
- `scripts/data/lm_dataset.py`: tokenized examples for MentalRoBERTa.

This split is important because traditional models should not import PyTorch,
and low-level feature loading should remain reusable.

