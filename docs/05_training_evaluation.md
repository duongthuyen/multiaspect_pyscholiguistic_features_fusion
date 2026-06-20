# Training And Evaluation

## Recommended CLI Entry Point

Prefer `scripts.main` for pipeline operations:

```powershell
python -m scripts.main preprocess
python -m scripts.main extract --input data/processed/train.csv --split train
python -m scripts.main combine
python -m scripts.main train fusion
python -m scripts.main train classical --model logistic_regression --features traditional
python -m scripts.main evaluate --split test
```

## Training Traditional Models

```powershell
python -m scripts.main train classical --model logistic_regression --features traditional --seed 42
python -m scripts.main train classical --model random_forest --features traditional --svd 300 --seed 42
python -m scripts.main train classical --model support_vector_machine --features traditional --svd 300 --seed 42
python -m scripts.main train classical --model xgboost --features traditional --svd 300 --seed 42
```

For Random Forest, SVM-RBF, and XGBoost, SVD keeps the TF-IDF block tractable.

## Training Fusion Models

```powershell
python -m scripts.main train fusion
python -m scripts.training.concat_train
python -m scripts.main run cross-attention
```

Multi-seed fusion runs:

```powershell
python -m scripts.main run multi-seed
```

## Evaluation Metrics

Primary metric:

- macro-F1

Supporting metrics:

- accuracy
- weighted-F1
- per-class precision, recall, F1
- confusion matrix
- branch/gate weights for interpretable fusion models

## Output Paths

Readable results:

```text
results/models/<paradigm>/<model>/evaluation/
results/models/<paradigm>/<model>/training/
results/models/<paradigm>/<model>/runs/seedN/
```

Reloadable model files:

```text
artifacts/models/<paradigm>/<model>/checkpoints/
artifacts/models/<paradigm>/<model>/runs/seedN/checkpoints/
```

