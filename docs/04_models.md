# Models

## MentalRoBERTa

The LM-based paradigm fine-tunes `mental/mental-roberta-base` with a
sequence-classification head.

Code:

- Model config and builder: `scripts/models/lm_based/mental_roberta.py`
- Dataset: `scripts/data/lm_dataset.py`
- Fine-tuning loop: `scripts/features/semantic/finetune_mental_roberta.py`
- Evaluation helper: `scripts/evaluation/lm_evaluate.py`

Outputs:

```text
results/models/lm_based/mental_roberta/
artifacts/models/lm_based/mental_roberta/
```

## Traditional Models

Traditional models use TF-IDF text features plus dense psycholinguistic
features. Semantic embeddings are intentionally excluded to keep the paradigm
separate from LM-based modeling.

Models:

- Logistic Regression
- Random Forest
- Support Vector Machine
- XGBoost

Code:

- Builders: `scripts/models/traditional/`
- Pipeline: `scripts/models/traditional/tfidf_pipeline.py`
- Dataset: `scripts/data/traditional_dataset.py`
- Training: `scripts/training/traditional.py`

## Fusion Models

Fusion models consume three branches:

- semantic: 768-dimensional MentalRoBERTa embedding
- affective: 34-dimensional affect vector
- handcrafted: 26-dimensional lexical/syntactic/structural vector

Implemented fusion architectures:

- `ConcatMLP`: concatenates all branches and classifies with an MLP.
- `GatedFusion`: projects branches and learns class-conditioned branch weights.
- `CrossAttentionFusion`: uses semantic features to attend over affective and
  handcrafted branches.

Code:

- Architectures: `scripts/models/fusion/`
- Training: `scripts/training/fusion_train.py`, `scripts/training/concat_train.py`,
  `scripts/training/runners/run_cross_attention.py`
- Evaluation: `scripts/evaluation/fusion_evaluate.py`

