# Reproducibility

## Environment

Recommended local environment:

```powershell
python -m venv amh_venv
.\amh_venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The NRC-VAD lexicon is required for VAD features:

```text
data/lexicons/NRC-VAD-Lexicon.txt
```

## End-To-End Order

1. Preprocess raw data.
2. Fine-tune MentalRoBERTa or restore the checkpoint.
3. Extract semantic, lexical, syntactic, structural, and affective features.
4. Train/evaluate traditional baselines.
5. Train/evaluate fusion models.
6. Run feature and branch analysis.
7. Generate report pack if needed.

## Commands

```powershell
python -m scripts.main preprocess

python -m scripts.features.semantic.finetune_mental_roberta --epochs 5 --batch 16 --lr 2e-5

python -m scripts.main extract --input data/processed/train.csv --split train
python -m scripts.main extract --input data/processed/val.csv --split val
python -m scripts.main extract --input data/processed/test.csv --split test

python -m scripts.main combine

python -m scripts.main train classical --model logistic_regression --features traditional --seed 42
python -m scripts.main train classical --model random_forest --features traditional --svd 300 --seed 42
python -m scripts.main train classical --model support_vector_machine --features traditional --svd 300 --seed 42
python -m scripts.main train classical --model xgboost --features traditional --svd 300 --seed 42

python -m scripts.main train fusion
python -m scripts.training.concat_train
python -m scripts.main run cross-attention
python -m scripts.main evaluate --split test

python -m scripts.main analyze-features --select
python -m scripts.main analyze-branches
python -m scripts.main report-pack
```

## Smoke Tests

```powershell
.\amh_venv\Scripts\python.exe -m pytest tests\data\test_feature_loader.py tests\features\semantic\test_finetune_mental_roberta.py tests\test_metrics.py tests\test_outputs.py -q
```

Some older model tests still need API/layout updates; see
[07_code_audit.md](07_code_audit.md).

