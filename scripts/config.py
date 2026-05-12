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
# are grouped by feature configuration, for example:
#   results/semantic/training/
#   results/semantic/evaluation/
#   results/fused/late_concat/training/
#   results/fused/gated/evaluation/

RESULTS_DIR     = ROOT_DIR / "results"
MODELS_DIR      = RESULTS_DIR / "models"
EVAL_DIR        = RESULTS_DIR / "evaluation"
PLOTS_DIR       = RESULTS_DIR / "plots"
LOGS_DIR        = RESULTS_DIR / "logs"


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
TOKENIZED_DIR         = ROBERTA_MODEL_DIR / "tokenized"
FINETUNED_ROBERTA_DIR = ROBERTA_MODEL_DIR / "finetuned"  # fine-tuned backbone without classification head
FUSION_MODEL_DIR      = MODELS_DIR / "fusion"
BASELINE_MODEL_DIR    = MODELS_DIR / "baselines"


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

MENTAL_ROBERTA_NAME = "mental/mental-roberta-base"  # domain-adapted variant
MAX_LENGTH          = 512       # Maximum token length RoBERTa accepts
BATCH_SIZE          = 32        # Number of samples per training step
LEARNING_RATE       = 2e-5      # Standard fine-tuning LR for transformers
NUM_EPOCHS          = 2         # Full passes over the training data
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
# TFIDF SETTINGS (for baseline only)
# =============================================================================

TFIDF_MAX_FEATURES  = 10000     # Vocabulary size limit
TFIDF_NGRAM_RANGE   = (1, 2)    # Unigrams and bigrams


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
# Two architectures: "concat" (primary) and "gated" (baseline for comparison).
# Selected at runtime via FUSION_TYPE.

FUSION_TYPE = "concat"

# Dimension-matched projections used by the concat fusion architecture.
# Each branch's projection size scales with its native dimensionality so the
# semantic branch doesn't dominate purely by dimension count.
SEMANTIC_PROJECTION_DIM    = 256
AFFECTIVE_PROJECTION_DIM   = 128
HANDCRAFTED_PROJECTION_DIM = 64

FUSION_DIM = (
    SEMANTIC_PROJECTION_DIM
    + AFFECTIVE_PROJECTION_DIM
    + HANDCRAFTED_PROJECTION_DIM
)

# Equal-projection size used by the gated fusion baseline.
GATED_PROJECTION_DIM = 256

# Training hyperparameters for the fusion model.
FUSION_LR     = 1e-3
FUSION_EPOCHS = 15

# Dropout rates.
BRANCH_DROPOUT     = 0.1
CLASSIFIER_DROPOUT = 0.2


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
# END-TO-END FUSION MODEL SETTINGS
# =============================================================================
# Trains RoBERTa backbone + fusion layers jointly in one pass.
# Two optimizer parameter groups with different LRs prevent the pre-trained
# backbone from being overwhelmed by randomly-initialised fusion gradients.

E2E_MODEL_DIR       = MODELS_DIR / "e2e"
E2E_BACKBONE_LR     = 2e-5   # same as original RoBERTa fine-tuning LR
E2E_FUSION_LR       = 1e-4   # 5× larger — fusion layers train from scratch
E2E_EPOCHS          = 5
E2E_BATCH_SIZE      = 16     # fits T4 (15 GB) at max_length=512
E2E_WEIGHT_DECAY    = 0.01
E2E_WARMUP_RATIO    = 0.1
E2E_GRAD_CLIP       = 1.0
E2E_DROPOUT         = 0.1


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42


