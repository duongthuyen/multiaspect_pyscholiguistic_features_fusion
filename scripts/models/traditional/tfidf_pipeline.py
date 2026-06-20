"""Traditional-paradigm pipeline: TF-IDF(text) [+ optional SVD] + dense -> classifier.

All preprocessing lives *inside* a scikit-learn Pipeline, fit on train only —
this structurally prevents leakage (TF-IDF vocabulary/IDF and the SVD basis are
learned from train and merely applied to val/test).

Two representations
-------------------
svd_components=None  (default, best for LINEAR models):
    ColumnTransformer[ TfidfVectorizer(text) | StandardScaler(with_mean=False)(dense) ]
    -> stays SPARSE (~20k + 60 dims). with_mean=False keeps sparsity on hstack.

svd_components=k  (needed to make RF / SVM-rbf / XGBoost tractable):
    text branch = TfidfVectorizer -> TruncatedSVD(k)  (= LSA, dense k dims)
    dense branch = passthrough
    -> everything DENSE (k + 60) -> StandardScaler (centering OK now).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import config
from scripts.data.traditional_dataset import TEXT_COLUMN, dense_feature_columns

# TF-IDF hyperparameters now live centrally in config.py (config.TFIDF_PARAMS).
TFIDF_PARAMS = config.TFIDF_PARAMS  # alias for backward compatibility


def build_preprocessor(
    n_dense: int,
    tfidf_params: dict | None = None,
    svd_components: int | None = None,
    seed: int = 42,
) -> ColumnTransformer:
    """ColumnTransformer for the text + dense blocks (with optional LSA on text)."""
    tfidf = TfidfVectorizer(**(tfidf_params or config.TFIDF_PARAMS))
    if svd_components:
        text_branch = Pipeline(
            [("tfidf", tfidf), ("svd", TruncatedSVD(n_components=svd_components, random_state=seed))]
        )
        dense_branch = "passthrough"
        sparse_threshold = 0.0  # SVD output is dense
    else:
        text_branch = tfidf
        dense_branch = StandardScaler(with_mean=False)
        sparse_threshold = 0.3
    return ColumnTransformer(
        transformers=[
            ("tfidf", text_branch, TEXT_COLUMN),
            ("dense", dense_branch, dense_feature_columns(n_dense)),
        ],
        sparse_threshold=sparse_threshold,
    )


def build_traditional_pipeline(
    estimator,
    n_dense: int,
    tfidf_params: dict | None = None,
    svd_components: int | None = None,
    seed: int = 42,
) -> Pipeline:
    """Full traditional pipeline: preprocessing (+ optional LSA) + classifier.

    With svd_components set, a StandardScaler is appended (everything is dense
    after SVD) so distance/kernel models like SVM-rbf are well-conditioned.
    """
    steps = [("features", build_preprocessor(n_dense, tfidf_params, svd_components, seed))]
    if svd_components:
        steps.append(("scale", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)
