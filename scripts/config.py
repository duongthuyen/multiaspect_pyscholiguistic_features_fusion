# scripts/config.py

from pathlib import Path

# =============================================================================
# ROOT
# =============================================================================
# Path(__file__) is the absolute path of this config.py.
# .parent is the folder containing it (scripts/).
# .parent again is the project root.
# Everything else is built relative to ROOT_DIR so the project works on any machine.

ROOT_DIR = Path(__file__).parent.parent


# =============================================================================
# DATA PATHS
# =============================================================================
# data/ holds all data: raw inputs, processed CSVs, and extracted features.
# Sub-folders separate read-only inputs (original, lexicons) from generated
# outputs (processed, features). Anything under data/ is pipeline plumbing.

DATA_DIR        = ROOT_DIR / "data"

# Read-only inputs
RAW_DIR         = DATA_DIR / "original"
LEXICONS_DIR    = DATA_DIR / "lexicons"

# Generated intermediate data
PROCESSED_DIR   = DATA_DIR / "processed"
FEATURES_DIR    = DATA_DIR / "features"

# Raw files (as downloaded — never written to)
RAW_TRAIN_PATH  = RAW_DIR / "both_train.csv"
RAW_VAL_PATH    = RAW_DIR / "both_val.csv"
RAW_TEST_PATH   = RAW_DIR / "both_test.csv"

# Processed files (output of preprocessing.py)
TRAIN_PATH      = PROCESSED_DIR / "train.csv"
VAL_PATH        = PROCESSED_DIR / "val.csv"
TEST_PATH       = PROCESSED_DIR / "test.csv"


# =============================================================================
# RESULTS PATHS
# =============================================================================
# results/ holds publication-relevant artifacts. Training and evaluation outputs
# are grouped by model family, for example:
#   results/gated_fusion/<variant>/training/
#   results/gated_fusion/<variant>/evaluation/

RESULTS_DIR     = ROOT_DIR / "results"
MODELS_DIR      = RESULTS_DIR / "models"
PLOTS_DIR       = RESULTS_DIR / "plots"


# =============================================================================
# FEATURE SUB-DIRECTORIES
# =============================================================================
# Each feature group has its own folder under data/features/.
# Within each group, sub-extractors save to their own files (e.g.,
# data/features/affective/goemotions.parquet, vad.parquet, vader.parquet).
# This makes ablation trivial: don't load a folder, drop the group.

SEMANTIC_FEATURES_DIR    = FEATURES_DIR / "semantic"
LEXICAL_FEATURES_DIR     = FEATURES_DIR / "lexical"
SYNTACTIC_FEATURES_DIR   = FEATURES_DIR / "syntactic"
STRUCTURAL_FEATURES_DIR  = FEATURES_DIR / "structural"
AFFECTIVE_FEATURES_DIR   = FEATURES_DIR / "affective"


# =============================================================================
# LEXICON PATHS
# =============================================================================
# Lexicons are read-only inputs. The NRC-VAD check is moved to a function
# (require_nrc_vad) so importing config doesn't crash when NRC-VAD is missing.
# Only modules that actually need NRC-VAD call require_nrc_vad() at init time.

ABSOLUTIST_LEXICON_PATH  = LEXICONS_DIR / "absolutist.txt"
NEGATION_LEXICON_PATH    = LEXICONS_DIR / "negation.txt"
MODAL_LEXICON_PATH       = LEXICONS_DIR / "modal.txt"
HEDGE_LEXICON_PATH       = LEXICONS_DIR / "hedge.txt"
DEATH_HARM_LEXICON_PATH  = LEXICONS_DIR / "death_harm.txt"
NRC_VAD_LEXICON_PATH     = LEXICONS_DIR / "NRC-VAD-Lexicon.txt"


def require_nrc_vad() -> None:
    """Raise if NRC-VAD lexicon is missing. Call from extractors that need it."""
    if not NRC_VAD_LEXICON_PATH.exists():
        raise FileNotFoundError(
            "Missing NRC-VAD lexicon. Download it from "
            "http://saifmohammad.com/WebPages/nrc-vad.html and place it at "
            f"{NRC_VAD_LEXICON_PATH}"
        )


# =============================================================================
# MODEL SUB-DIRECTORIES
# =============================================================================

ROBERTA_MODEL_DIR     = MODELS_DIR / "roberta"
FINETUNED_ROBERTA_DIR = ROBERTA_MODEL_DIR / "finetuned"  # fine-tuned backbone without classification head


# =============================================================================
# COLUMN NAMES
# =============================================================================
# Centralizing column names means a CSV column rename only requires a one-line
# change here, not edits across every file that reads the data.

RAW_TITLE_COL   = "title"
RAW_POST_COL    = "post"
TEXT_COL        = "text"        # Merged column created in preprocessing
LABEL_COL       = "class_id"    # Integer label used for model training
CLASS_NAME_COL  = "class_name"  # Human-readable label used for analysis/plots


# =============================================================================
# CLASS LABELS
# =============================================================================

NUM_LABELS = 6

ID_TO_CLASS = {
    0: "ADHD",
    1: "Anxiety",
    2: "Bipolar",
    3: "Depression",
    4: "PTSD",
    5: "None",
}

CLASS_TO_ID = {v: k for k, v in ID_TO_CLASS.items()}


# =============================================================================
# PREPROCESSING SETTINGS
# =============================================================================

# Reads naturally as "Title: post body..." when title and body are merged.
MERGE_SEPARATOR = ": "


# =============================================================================
# ROBERTA SETTINGS
# =============================================================================

MENTAL_ROBERTA_NAME  = "mental/mental-roberta-base"  # domain-adapted variant
MAX_LENGTH           = 512   # Maximum token length RoBERTa accepts
ROBERTA_BATCH_SIZE   = 16   # Batch size for MentalRoBERTa fine-tuning (GPU-sensitive)
FUSION_BATCH_SIZE    = 32   # Batch size for gated fusion DataLoaders
LEARNING_RATE        = 2e-5  # Standard fine-tuning LR for transformers
NUM_EPOCHS          = 5         # Full passes over the training data
WEIGHT_DECAY        = 0.01      # Regularization to prevent overfitting
WARMUP_RATIO        = 0.1       # Fraction of total steps used for LR warm-up
GRAD_CLIP           = 1.0       # Gradient norm clipping threshold


# =============================================================================
# EMOTION MODEL SETTINGS
# =============================================================================

EMOTION_MODEL_NAME  = "SamLowe/roberta-base-go_emotions"
EMOTION_BATCH_SIZE  = 16


# =============================================================================
# SENTENCE EMBEDDING MODEL (for structural coherence)
# =============================================================================

SENTENCE_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COHERENCE_BREAK_THRESHOLD = 0.3   # cosine sim below this counts as a "break"


# =============================================================================
# SPACY MODEL
# =============================================================================

SPACY_MODEL = "en_core_web_sm"


# =============================================================================
# FEATURE FRAMEWORK
# =============================================================================
# Five groups based on Lagutina stylometric taxonomy with FTD clinical anchoring.
# Total handcrafted = 11 + 8 + 7 + 34 = 60 dims.
# Plus 768 semantic dims = 828 total.

FEATURE_GROUPS = ["semantic", "lexical", "syntactic", "structural", "affective"]

FEATURE_DIMS = {
    "semantic"  : 768,
    "lexical"   : 11,
    "syntactic" : 8,
    "structural": 7,
    "affective" : 34,
}

SEMANTIC_DIM    = FEATURE_DIMS["semantic"]
LEXICAL_DIM     = FEATURE_DIMS["lexical"]
SYNTACTIC_DIM   = FEATURE_DIMS["syntactic"]
STRUCTURAL_DIM  = FEATURE_DIMS["structural"]
AFFECTIVE_DIM   = FEATURE_DIMS["affective"]
HANDCRAFTED_DIM = LEXICAL_DIM + SYNTACTIC_DIM + STRUCTURAL_DIM
TOTAL_FEATURE_DIM = sum(FEATURE_DIMS.values())


# =============================================================================
# FUSION MODEL SETTINGS
# =============================================================================
# Output layout for the gated fusion model.
GATED_FUSION_OUTPUT_DIR = "gated_fusion"
GATED_FUSION_INPUT_CONFIG = "fused"
DEFAULT_GATED_VARIANT = "gated_fusion"

# Architecture hyperparameters
GATED_PROJECTION_DIM = 256
GATED_GATE_HIDDEN_DIM = 128
GATED_HANDCRAFTED_DROPOUT = 0.4
GATED_AUX_WEIGHT = 0.3          # weight for auxiliary semantic classification loss
GATED_DIVERSITY_WEIGHT = 0.01   # weight for gate-diversity penalty (lambda_div)

# Training defaults
FUSION_LR = 5e-4
FUSION_EPOCHS = 20
FUSION_LABEL_SMOOTHING = 0.1
FUSION_GATE_WEIGHT_DECAY = 1e-4
FUSION_EARLY_STOPPING_PATIENCE = 2

GATED_FUSION_DEFAULTS = {
    "model": DEFAULT_GATED_VARIANT,
    "input_config": GATED_FUSION_INPUT_CONFIG,
    "projection_dim": GATED_PROJECTION_DIM,
    "gate_hidden_dim": GATED_GATE_HIDDEN_DIM,
    "handcrafted_dropout": GATED_HANDCRAFTED_DROPOUT,
    "aux_weight": GATED_AUX_WEIGHT,
    "diversity_weight": GATED_DIVERSITY_WEIGHT,
    "epochs": FUSION_EPOCHS,
    "lr": FUSION_LR,
    "batch_size": FUSION_BATCH_SIZE,
    "seed": 42,
    "label_smoothing": FUSION_LABEL_SMOOTHING,
    "gate_weight_decay": FUSION_GATE_WEIGHT_DECAY,
    "early_stopping_patience": FUSION_EARLY_STOPPING_PATIENCE,
}


def get_gated_fusion_config(
    variant: str | None = None,
    overrides: dict | None = None,
) -> dict:
    """Return the full config dict for the gated fusion model.

    Parameters
    ----------
    variant : str, optional
        Must be ``"gated_fusion"`` or None (defaults to ``"gated_fusion"``).
    overrides : dict, optional
        Any keys here override the defaults.
    """
    selected = variant or DEFAULT_GATED_VARIANT
    if selected != DEFAULT_GATED_VARIANT:
        raise ValueError(
            f"Unknown gated fusion variant: {selected!r}. "
            f"Only 'gated_fusion' is supported."
        )
    cfg = dict(GATED_FUSION_DEFAULTS)
    cfg["model"] = selected
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# =============================================================================
# CLASSICAL MODEL SETTINGS
# =============================================================================

LOGISTIC_REGRESSION_MAX_ITER = 2000

RANDOM_FOREST_N_ESTIMATORS = 500
RANDOM_FOREST_MAX_DEPTH = None

SVM_C = 1.0
SVM_KERNEL = "rbf"

XGBOOST_N_ESTIMATORS = 500
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.05




# =============================================================================
# REPRODUCIBILITY
# ============================================================================================

SEED = 42
