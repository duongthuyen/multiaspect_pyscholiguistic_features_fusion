# scripts/analysis/feature_statistics.py
"""
Per-class feature profile analysis for all interpretable feature groups.

Semantic features (768-dim MentalRoBERTa embeddings) are excluded — individual
dimensions have no linguistic interpretation and cannot be meaningfully plotted.
The four remaining groups (affective, lexical, syntactic, structural) contain
named features that can be read directly off a heatmap.

Normalisation (scaled mode):
    A StandardScaler is fit on the full training split before computing
    per-class mean profiles.  This puts all features on the same scale
    (zero-mean, unit-variance) so that unbounded features such as MTLD or
    readability grades do not visually swamp bounded ratios in the heatmap.
    The scaler is fit once per group and applied to the same training data
    used for profiling.

Raw mode:
    No scaling is applied — actual feature values are used throughout.
    Violin plots show the full per-class distribution of each feature so that
    spread, skew, and outliers are visible alongside the class mean.
    Each sub-feature file (e.g., affective/goemotions) gets its own plot so
    that features sharing a natural scale are always compared together.

Output layout (one subdirectory per group):
    results/plots/feature_statistics/{group}/           ← scaled heatmap
        {group}_heatmap.png
        {group}_profile_raw.csv
        {group}_profile_normalized.csv
    results/plots/feature_statistics_raw/{group}/       ← raw distributions
        {sub_name}_violin.png
        {group}_raw_mean.csv
        {group}_mean_eta2.csv
"""

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from scripts.config import (
    AFFECTIVE_FEATURES_DIR,
    CLASS_NAME_COL,
    LEXICAL_FEATURES_DIR,
    PLOTS_DIR,
    STRUCTURAL_FEATURES_DIR,
    SYNTACTIC_FEATURES_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)

logger = logging.getLogger(__name__)
sns.set_theme(style="white")


# =============================================================================
# REGISTRIES
# =============================================================================
# Only interpretable groups — semantic (768-dim embeddings) is excluded.

INTERPRETABLE_GROUPS = ["affective", "lexical", "syntactic", "structural"]

GROUP_BASE_DIRS = {
    "affective":  AFFECTIVE_FEATURES_DIR,
    "lexical":    LEXICAL_FEATURES_DIR,
    "syntactic":  SYNTACTIC_FEATURES_DIR,
    "structural": STRUCTURAL_FEATURES_DIR,
}

# Sub-feature files that exist in each group's split directory
GROUP_SUBFEATURES = {
    "affective":  ["goemotions", "vad", "vader"],
    "lexical":    ["diversity", "word_rates", "pronouns", "punctuation"],
    "syntactic":  ["complexity", "pos_ratios", "readability"],
    "structural": ["coherence", "tense"],
}

# Human-readable column names — must match the order features are concatenated
SUB_FEATURE_NAMES = {
    "affective": {
        "goemotions": [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral",
        ],
        "vad":   ["mean_valence", "mean_arousal", "mean_dominance"],
        "vader": ["sentiment_mean", "sentiment_std", "sentiment_range"],
    },
    "lexical": {
        "diversity":   ["mtld"],
        "word_rates":  ["death_harm_rate", "absolutist_rate", "negation_rate",
                        "modal_rate", "hedge_rate"],
        "pronouns":    ["1p_singular_rate", "1p_plural_rate", "2p_rate"],
        "punctuation": ["question_mark_rate", "ellipsis_rate"],
    },
    "syntactic": {
        "complexity": ["mean_dep_distance", "mean_tree_depth", "mean_sent_length"],
        "pos_ratios": ["adjective_ratio", "adverb_ratio", "pronoun_ratio"],
        "readability": ["flesch_kincaid_grade", "gunning_fog"],
    },
    "structural": {
        "coherence": ["mean_coherence", "std_coherence", "topic_drift", "break_rate"],
        "tense":     ["past_ratio", "present_ratio", "future_ratio"],
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

def _feature_path(group: str, sub_name: str, split: str) -> Path:
    """Return the parquet path for one sub-feature, split-aware."""
    base = GROUP_BASE_DIRS[group]
    candidate = base / split / f"{sub_name}.parquet"
    if candidate.exists():
        return candidate
    # Fallback: some older extractions saved without split subfolder
    fallback = base / f"{sub_name}.parquet"
    return fallback


def load_group_features(group: str, split: str = "train") -> pd.DataFrame:
    """
    Load all sub-features for *group* / *split*, concatenate horizontally,
    and assign proper column names.

    Returns a DataFrame with shape (n_samples, n_features_in_group).
    """
    frames = []
    for sub_name in GROUP_SUBFEATURES[group]:
        path = _feature_path(group, sub_name, split)
        if not path.exists():
            logger.warning("Missing parquet — skipping: %s", path)
            continue

        df  = pd.read_parquet(path)
        mat = np.asarray(df["features"].tolist(), dtype=np.float32)

        names = SUB_FEATURE_NAMES[group].get(sub_name)
        if names is None or len(names) != mat.shape[1]:
            # Fallback to generic names if something is off
            names = [f"{sub_name}_{i}" for i in range(mat.shape[1])]

        frames.append(pd.DataFrame(mat, columns=names))
        logger.info("  Loaded %s/%s: %s", group, sub_name, mat.shape)

    if not frames:
        raise RuntimeError(
            f"No feature files found for group '{group}' / split '{split}'. "
            "Run feature extraction first."
        )
    return pd.concat(frames, axis=1)


def load_labels(split: str = "train") -> pd.Series:
    path_map = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}
    return pd.read_csv(path_map[split])[CLASS_NAME_COL]


# =============================================================================
# NORMALISATION + PROFILING
# =============================================================================

def standardize_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a StandardScaler on *features_df* and return the scaled DataFrame.

    Each column becomes zero-mean and unit-variance across all training rows.
    This ensures that MTLD (~50-300) and readability grades (~0-20) do not
    visually dominate bounded ratios (~0-1) in the heatmap.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_df.values)
    return pd.DataFrame(scaled, columns=features_df.columns)


def compute_class_profile(features_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Per-class mean feature values.  Returns (n_classes, n_features)."""
    combined = features_df.copy()
    combined["__class__"] = labels.values
    return combined.groupby("__class__").mean().sort_index()


def min_max_normalize(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Scale each feature column to [0, 1] across classes so the heatmap shows
    'which class is highest / lowest on this feature' rather than raw magnitude.
    Constant columns (no variation across classes) are set to 0.5.
    """
    col_min   = profile.min()
    col_max   = profile.max()
    col_range = col_max - col_min
    normalized = profile.copy()
    for col in profile.columns:
        if col_range[col] == 0:
            normalized[col] = 0.5
        else:
            normalized[col] = (profile[col] - col_min[col]) / col_range[col]
    return normalized


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_heatmap(normalized_profile: pd.DataFrame, group: str, save_dir) -> None:
    """Heatmap: rows = features, columns = classes, values in [0, 1]."""
    plot_data = normalized_profile.T          # transpose → features as rows
    n_features = plot_data.shape[0]
    fig_height = max(6, n_features * 0.40 + 2)

    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.heatmap(
        plot_data,
        ax        = ax,
        annot     = True,
        fmt       = ".2f",
        cmap      = "YlGnBu",
        cbar      = True,
        cbar_kws  = {"label": "Normalised class mean", "shrink": 0.7},
        linewidths = 0.5,
        linecolor  = "white",
        annot_kws  = {"size": 9},
    )
    ax.set_title(
        f"{group.title()} Feature Fingerprint by Mental Health Condition",
        fontsize=14, pad=20,
    )
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Feature",   fontsize=12)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()

    path = save_dir / f"{group}_heatmap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Heatmap → %s", path)


# =============================================================================
# CSV OUTPUT
# =============================================================================

def save_profile_tables(
    profile: pd.DataFrame,
    normalized_profile: pd.DataFrame,
    group: str,
    save_dir,
) -> None:
    raw_path  = save_dir / f"{group}_profile_raw.csv"
    norm_path = save_dir / f"{group}_profile_normalized.csv"
    profile.to_csv(raw_path)
    normalized_profile.to_csv(norm_path)
    logger.info("  Raw profile       → %s", raw_path)
    logger.info("  Normalised profile→ %s", norm_path)


def report_top_features(normalized_profile: pd.DataFrame, group: str, top_k: int = 3) -> None:
    logger.info("\n  Top %d features per class (%s):", top_k, group)
    for cls in normalized_profile.index:
        top = normalized_profile.loc[cls].sort_values(ascending=False).head(top_k)
        logger.info("    %-12s %s", cls,
                    ", ".join(f"{f} ({v:.2f})" for f, v in top.items()))


# =============================================================================
# PIPELINE
# =============================================================================

def analyze_group(group: str, base_save_dir) -> None:
    """Full analysis pipeline for one feature group."""
    logger.info("\n%s\nAnalysing group: %s\n%s", "=" * 60, group, "=" * 60)

    # Per-group subdirectory
    save_dir = base_save_dir / group
    save_dir.mkdir(parents=True, exist_ok=True)

    features_df = load_group_features(group, split="train")
    labels      = load_labels(split="train")

    if len(features_df) != len(labels):
        raise RuntimeError(
            f"Row count mismatch in {group}: "
            f"features={len(features_df)}, labels={len(labels)}"
        )

    # Normalise raw features before profiling so scales are comparable
    scaled_df = standardize_features(features_df)

    profile            = compute_class_profile(scaled_df, labels)
    normalized_profile = min_max_normalize(profile)

    plot_heatmap(normalized_profile, group, save_dir)
    save_profile_tables(profile, normalized_profile, group, save_dir)
    report_top_features(normalized_profile, group)


def run_feature_statistics() -> None:
    """
    Run analysis for all interpretable groups (affective, lexical, syntactic,
    structural).  Results land in results/plots/feature_statistics/{group}/.
    """
    base_save_dir = PLOTS_DIR / "feature_statistics"
    base_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Feature statistics analysis — %d groups", len(INTERPRETABLE_GROUPS))
    logger.info("=" * 60)

    for group in INTERPRETABLE_GROUPS:
        try:
            analyze_group(group, base_save_dir)
        except Exception as exc:
            logger.error("Failed to analyse %s: %s", group, exc)

    logger.info("\n%s", "=" * 60)
    logger.info("Done. Results in: %s", base_save_dir)
    logger.info("=" * 60)


# =============================================================================
# RAW DATA VISUALISATION
# =============================================================================

_CLASS_ORDER = ["ADHD", "Anxiety", "Bipolar", "Depression", "PTSD", "None"]
_PALETTE = "Set2"


def _load_subfeature_raw(group: str, sub_name: str, split: str = "train") -> pd.DataFrame:
    """Load one sub-feature file and return a DataFrame with named columns."""
    path = _feature_path(group, sub_name, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    df = pd.read_parquet(path)
    mat = np.asarray(df["features"].tolist(), dtype=np.float32)
    names = SUB_FEATURE_NAMES[group].get(sub_name)
    if names is None or len(names) != mat.shape[1]:
        names = [f"{sub_name}_{i}" for i in range(mat.shape[1])]
    return pd.DataFrame(mat, columns=names)


def _eta_squared(sub_df: pd.DataFrame, combined: pd.DataFrame) -> pd.Series:
    """Eta-squared (η²) per feature column via one-way ANOVA decomposition."""
    grand_means = sub_df.mean()
    ss_between  = pd.Series(0.0, index=sub_df.columns)
    for _, grp in combined.groupby("__class__"):
        n_cls = len(grp)
        cls_means = grp[sub_df.columns].mean()
        ss_between += n_cls * (cls_means - grand_means) ** 2
    ss_total = ((sub_df - grand_means) ** 2).sum()
    return (ss_between / ss_total.replace(0, np.nan)).fillna(0.0)


def plot_raw_heatmap(
    sub_df: pd.DataFrame,
    labels: pd.Series,
    title: str,
    save_path: Path,
) -> None:
    """
    Heatmap of raw (unscaled) per-class means with η².

    Left panel: cells annotated as "mean\\n(±std)".
    Right panel: η² (eta-squared) effect size per feature — proportion of
    total variance explained by class membership (small ≈ 0.01, medium ≈ 0.06,
    large ≥ 0.14).
    """
    combined = sub_df.copy()
    combined["__class__"] = labels.values

    present = set(labels.unique())
    class_order = [c for c in _CLASS_ORDER if c in present] or sorted(present)

    grouped       = combined.groupby("__class__")
    mean_profile  = grouped.mean().loc[class_order].T   # features × classes
    eta2_series   = _eta_squared(sub_df, combined)
    feature_order = eta2_series.sort_values(ascending=False).index
    mean_profile = mean_profile.loc[feature_order]
    eta2_series = eta2_series.loc[feature_order]

    # Adaptive annotation format based on mean magnitude
    max_abs = np.abs(mean_profile.values).max()
    if max_abs < 0.01:
        fmt = ".4f"
    elif max_abs < 0.1:
        fmt = ".3f"
    elif max_abs < 10:
        fmt = ".2f"
    else:
        fmt = ".1f"

    n_features = mean_profile.shape[0]
    n_classes  = len(class_order)
    fig_height = max(5, n_features * 0.80 + 2)

    has_negative = (mean_profile.values < 0).any()
    mean_cmap = "RdBu_r" if has_negative else "YlOrRd"
    center    = 0.0 if has_negative else None

    fig = plt.figure(figsize=(n_classes * 3.4 + 2.5, fig_height))
    gs  = fig.add_gridspec(1, 2, width_ratios=[n_classes, 1.3], wspace=0.12)
    ax_mean = fig.add_subplot(gs[0])
    ax_eta  = fig.add_subplot(gs[1])

    # --- Mean panel ---
    sns.heatmap(
        mean_profile,
        ax         = ax_mean,
        annot      = True,
        fmt        = fmt,
        cmap       = mean_cmap,
        center     = center,
        linewidths = 0.5,
        linecolor  = "white",
        cbar_kws   = {"label": "Raw class mean", "shrink": 0.6},
        annot_kws  = {"size": 8},
    )
    ax_mean.set_title(title, fontsize=13, pad=16)
    ax_mean.set_xlabel("Class",   fontsize=11)
    ax_mean.set_ylabel("Feature", fontsize=11)
    ax_mean.tick_params(axis="x", rotation=30, labelsize=10)
    ax_mean.tick_params(axis="y", rotation=0,  labelsize=9)

    # --- η² panel ---
    eta_vmax = max(float(eta2_series.max()), 0.05)
    sns.heatmap(
        eta2_series.to_frame("η²"),
        ax         = ax_eta,
        annot      = True,
        fmt        = ".3f",
        cmap       = "Greens",
        vmin       = 0,
        vmax       = eta_vmax,
        linewidths = 0.5,
        linecolor  = "white",
        cbar_kws   = {"label": "η²  (small≥.01  med≥.06  large≥.14)", "shrink": 0.6},
        annot_kws  = {"size": 8},
    )
    ax_eta.set_title("Effect size (η²)", fontsize=11, pad=16)
    ax_eta.set_xlabel("")
    ax_eta.set_ylabel("")
    ax_eta.tick_params(axis="y", labelleft=False, left=False)
    ax_eta.tick_params(axis="x", rotation=30, labelsize=10)

    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info("  Raw heatmap → %s", save_path)


def plot_group_heatmap_combined(
    group: str,
    labels: pd.Series,
    save_path: Path,
) -> None:
    """
    One figure per group: all sub-features stacked vertically, each block with
    its own color scale so features on different scales remain readable.
    Two-panel layout per block: mean | η².
    """
    present     = set(labels.unique())
    class_order = [c for c in _CLASS_ORDER if c in present] or sorted(present)
    n_classes   = len(class_order)

    # Load stats for every sub-feature
    sub_stats   = {}   # sub_name → (mean_profile, eta2_series)
    row_heights = []
    for sub_name in GROUP_SUBFEATURES[group]:
        try:
            sub_df = _load_subfeature_raw(group, sub_name, split="train")
        except FileNotFoundError:
            logger.warning("Skipping %s/%s — file not found", group, sub_name)
            continue
        combined  = sub_df.copy()
        combined["__class__"] = labels.values
        grouped   = combined.groupby("__class__")
        mean_p    = grouped.mean().loc[class_order].T
        eta2      = _eta_squared(sub_df, combined)
        feature_order = eta2.sort_values(ascending=False).index
        mean_p = mean_p.loc[feature_order]
        eta2 = eta2.loc[feature_order]
        sub_stats[sub_name] = (mean_p, eta2)
        row_heights.append(mean_p.shape[0])

    if not sub_stats:
        return

    n_rows     = len(sub_stats)
    total_feat = sum(row_heights)
    fig_height = max(6, total_feat * 0.85 + n_rows * 0.6 + 1.5)
    fig_width  = n_classes * 3.4 + 2.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = fig.add_gridspec(
        n_rows, 2,
        height_ratios = row_heights,
        width_ratios  = [n_classes, 1.3],
        hspace        = 0.55,
        wspace        = 0.12,
    )

    for row_idx, (sub_name, (mean_p, eta2)) in enumerate(sub_stats.items()):
        ax_mean = fig.add_subplot(gs[row_idx, 0])
        ax_eta  = fig.add_subplot(gs[row_idx, 1])

        is_last = (row_idx == n_rows - 1)

        max_abs   = np.abs(mean_p.values).max()
        fmt       = ".4f" if max_abs < 0.01 else ".3f" if max_abs < 0.1 else ".2f" if max_abs < 10 else ".1f"
        has_neg   = (mean_p.values < 0).any()
        mean_cmap = "RdBu_r" if has_neg else "YlOrRd"
        center    = 0.0 if has_neg else None

        # Mean
        sns.heatmap(
            mean_p, ax=ax_mean, annot=True, fmt=fmt,
            cmap=mean_cmap, center=center,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Mean", "shrink": 0.7},
            annot_kws={"size": 8},
            xticklabels=is_last,
        )
        ax_mean.set_ylabel(f"[{sub_name}]\nFeature", fontsize=8, labelpad=6)
        ax_mean.tick_params(axis="y", rotation=0, labelsize=8)
        ax_mean.set_xlabel("Class" if is_last else "", fontsize=9)
        ax_mean.tick_params(axis="x", rotation=30, labelsize=9)
        if row_idx == 0:
            ax_mean.set_title(f"{group.title()} — Raw Class Means", fontsize=12, pad=10)

        # η²
        eta_vmax = max(float(eta2.max()), 0.05)
        sns.heatmap(
            eta2.to_frame("η²"), ax=ax_eta, annot=True, fmt=".3f",
            cmap="Greens", vmin=0, vmax=eta_vmax,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "η²  (small≥.01  med≥.06  large≥.14)", "shrink": 0.7},
            annot_kws={"size": 8},
            xticklabels=is_last,
        )
        ax_eta.set_ylabel("")
        ax_eta.tick_params(axis="y", labelleft=False, left=False)
        ax_eta.tick_params(axis="x", rotation=30, labelsize=9)
        if row_idx == 0:
            ax_eta.set_title("Effect size (η²)", fontsize=11, pad=10)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Group heatmap → %s", save_path)


def plot_raw_violin(
    features_df: pd.DataFrame,
    labels: pd.Series,
    title: str,
    save_path: Path,
) -> None:
    """
    Violin plot grid: one subplot per feature column, x=class, y=raw value.
    Shows the full per-class distribution — spread, skew, and outliers — at
    actual feature scale with no normalization applied.
    """
    cols = features_df.columns.tolist()
    n = len(cols)
    ncols = min(7, n)
    nrows = math.ceil(n / ncols)

    combined = features_df.copy()
    combined["class"] = labels.values

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.8, nrows * 4.0),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    present = set(combined["class"].unique())
    class_order = [c for c in _CLASS_ORDER if c in present] or sorted(present)

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        sns.violinplot(
            data=combined,
            x="class",
            y=col,
            order=class_order,
            hue="class",
            hue_order=class_order,
            palette=_PALETTE,
            legend=False,
            ax=ax,
            inner="box",
            cut=0,
            linewidth=0.8,
        )
        ax.set_title(col, fontsize=9, pad=5)
        ax.set_xlabel("")
        ax.set_ylabel("Raw value", fontsize=8)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.3g}")
        )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Violin → %s", save_path)


def save_raw_summary(
    features_df: pd.DataFrame,
    labels: pd.Series,
    group: str,
    save_dir: Path,
) -> None:
    """Save per-feature class means with eta-squared, sorted by effect size."""
    combined = features_df.copy()
    combined["__class__"] = labels.values
    present = set(labels.unique())
    class_order = [c for c in _CLASS_ORDER if c in present] or sorted(present)
    mean_profile = combined.groupby("__class__").mean().loc[class_order].T
    eta2_series = _eta_squared(features_df, combined).sort_values(ascending=False)
    summary = mean_profile.loc[eta2_series.index].copy()
    summary.insert(0, "eta2", eta2_series.values)

    summary.to_csv(save_dir / f"{group}_mean_eta2.csv", index_label="feature")
    logger.info("  Raw summary CSVs → %s", save_dir)


def analyze_group_raw(group: str, base_save_dir: Path) -> None:
    """Raw class-mean/effect-size plots for every sub-feature file in one group."""
    logger.info("\n%s\nRaw analysis: %s\n%s", "=" * 60, group, "=" * 60)

    save_dir = base_save_dir / group
    save_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(split="train")
    all_frames = []

    for sub_name in GROUP_SUBFEATURES[group]:
        try:
            sub_df = _load_subfeature_raw(group, sub_name, split="train")
        except FileNotFoundError as exc:
            logger.warning("Skipping %s/%s — %s", group, sub_name, exc)
            continue

        if len(sub_df) != len(labels):
            raise RuntimeError(
                f"Row count mismatch {group}/{sub_name}: "
                f"features={len(sub_df)}, labels={len(labels)}"
            )

        all_frames.append(sub_df)
        base_title = f"{group.title()} / {sub_name}"
        plot_raw_heatmap(
                sub_df, labels,
            f"{base_title} — Raw Class Means",
            save_dir / f"{sub_name}_heatmap_raw.png",
        )
        if False:
            plot_raw_violin(
            sub_df, labels,
            f"{base_title} — Raw Distributions by Class",
                save_dir / f"{sub_name}_violin.png",
            )

    if all_frames:
        full_df = pd.concat(all_frames, axis=1)
        save_raw_summary(full_df, labels, group, save_dir)

    plot_group_heatmap_combined(group, labels, save_dir / f"{group}_heatmap_combined.png")


def run_feature_statistics_raw() -> None:
    """
    Raw-data visualization for all interpretable groups.
    No scaling is applied; plots show per-class means plus eta-squared effect size.
    Results land in results/plots/feature_statistics_raw/{group}/.
    """
    base_save_dir = PLOTS_DIR / "feature_statistics_raw"
    base_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Raw feature statistics — %d groups", len(INTERPRETABLE_GROUPS))
    logger.info("=" * 60)

    for group in INTERPRETABLE_GROUPS:
        try:
            analyze_group_raw(group, base_save_dir)
        except Exception as exc:
            logger.error("Failed: %s: %s", group, exc)

    logger.info("\n%s", "=" * 60)
    logger.info("Done. Results in: %s", base_save_dir)
    logger.info("=" * 60)


# =============================================================================
# EFFECT SIZE FEATURE SELECTION
# =============================================================================

# η² thresholds from Cohen (1988) — widely used in NLP and psychology research.
# See: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
#      (2nd ed.). Chapter 8 — f² conversions map to η² via η² = f²/(1+f²).
_ETA2_THRESHOLDS = {
    "DROP":        (0.00, 0.01),   # η² < 0.01  — negligible class signal
    "KEEP":        (0.01, 0.06),   # η² ∈ [0.01, 0.06) — small effect
    "STRONG_KEEP": (0.06, 1.01),   # η² ≥ 0.06  — medium or large effect
}


def _classify_eta2(eta2: float) -> str:
    """Map an η² value to a keep/drop recommendation string."""
    if eta2 < 0.01:
        return "DROP"
    elif eta2 < 0.06:
        return "KEEP"
    else:
        return "STRONG_KEEP"


def compute_effect_sizes(split: str = "train") -> pd.DataFrame:
    """
    Compute η² for every named feature across all interpretable groups.

    Inputs
    ------
    split : str
        Data split to use (default: "train").  η² is always estimated on the
        training set to avoid data leakage into a selection decision.

    Returns
    -------
    pd.DataFrame with columns:
        feature       — feature name (str)
        group         — feature group ("affective", "lexical", …)
        sub_name      — sub-extractor name (e.g. "vader", "diversity")
        eta2          — η² value in [0, 1]
        recommendation — "DROP" / "KEEP" / "STRONG_KEEP"
        effect_label  — "negligible" / "small" / "medium" / "large"
    sorted by η² descending.
    """
    labels = load_labels(split)
    rows: list[dict] = []

    for group in INTERPRETABLE_GROUPS:
        for sub_name in GROUP_SUBFEATURES[group]:
            try:
                sub_df = _load_subfeature_raw(group, sub_name, split)
            except FileNotFoundError:
                logger.warning("Skipping %s/%s — file not found", group, sub_name)
                continue

            if len(sub_df) != len(labels):
                logger.warning(
                    "Row mismatch %s/%s (%d vs %d) — skipping",
                    group, sub_name, len(sub_df), len(labels),
                )
                continue

            # _eta_squared() requires a combined DataFrame with a '__class__' column.
            combined = sub_df.copy()
            combined["__class__"] = labels.values

            eta2_series = _eta_squared(sub_df, combined)

            for feature_name, eta2_val in eta2_series.items():
                # Effect label — descriptive string alongside the numeric value
                if eta2_val < 0.01:
                    effect_label = "negligible"
                elif eta2_val < 0.06:
                    effect_label = "small"
                elif eta2_val < 0.14:
                    effect_label = "medium"
                else:
                    effect_label = "large"

                rows.append({
                    "feature":        feature_name,
                    "group":          group,
                    "sub_name":       sub_name,
                    "eta2":           round(float(eta2_val), 6),
                    "recommendation": _classify_eta2(float(eta2_val)),
                    "effect_label":   effect_label,
                })

    df = pd.DataFrame(rows).sort_values("eta2", ascending=False).reset_index(drop=True)
    return df


def plot_effect_size_bar(report_df: pd.DataFrame, save_path: Path) -> None:
    """
    Horizontal bar chart of η² per feature, color-coded by recommendation.

    Color scheme:
        DROP        → light red / salmon
        KEEP        → steel blue
        STRONG_KEEP → forest green

    The vertical reference lines mark the small (0.01) and medium (0.06)
    thresholds so the reader can visually locate the decision boundaries.
    """
    color_map = {
        "DROP":        "#e57373",   # light red
        "KEEP":        "#5b9bd5",   # steel blue
        "STRONG_KEEP": "#4caf50",   # forest green
    }

    df = report_df.copy()
    df["color"] = df["recommendation"].map(color_map)
    df["label"] = df["feature"] + "  [" + df["group"] + "]"

    n = len(df)
    fig_height = max(6, n * 0.32 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bars = ax.barh(
        y     = range(n),
        width = df["eta2"].values,
        color = df["color"].values,
        edgecolor = "white",
        linewidth = 0.4,
        height = 0.75,
    )

    # Decision boundary lines
    ax.axvline(0.01, color="gray",  linestyle="--", linewidth=1.0, label="small (0.01)")
    ax.axvline(0.06, color="black", linestyle="--", linewidth=1.0, label="medium (0.06)")
    ax.axvline(0.14, color="black", linestyle=":",  linewidth=1.0, label="large (0.14)")

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["label"].values, fontsize=7)
    ax.invert_yaxis()   # highest η² at the top

    ax.set_xlabel("η² (eta-squared effect size)", fontsize=11)
    ax.set_title("Feature Selection by Effect Size (η²)", fontsize=12, pad=10)

    # Annotate η² values on bars
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["eta2"] + 0.001, i,
            f"{row['eta2']:.4f}",
            va="center", ha="left", fontsize=6.5, color="dimgray",
        )

    # Legend: colour patches for recommendation categories + threshold lines
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_map["STRONG_KEEP"], label="STRONG_KEEP  (η² ≥ 0.06)"),
        Patch(facecolor=color_map["KEEP"],        label="KEEP  (0.01 ≤ η² < 0.06)"),
        Patch(facecolor=color_map["DROP"],        label="DROP  (η² < 0.01)"),
        plt.Line2D([0], [0], color="gray",  linestyle="--", linewidth=1.2, label="threshold 0.01"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="threshold 0.06"),
        plt.Line2D([0], [0], color="black", linestyle=":",  linewidth=1.2, label="threshold 0.14"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="gray")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info("  Effect size bar chart → %s", save_path)


def run_effect_size_selection(split: str = "train") -> None:
    """
    Compute η² for every named feature, apply Cohen (1988) thresholds, and
    export a keep/drop report.

    Algorithm
    ---------
    1. Load each sub-feature parquet from *split* (default: training set).
    2. Compute η² = SS_between / SS_total via one-way ANOVA decomposition
       (reuses _eta_squared() already present in this module).
    3. Classify each feature:
           DROP        if η² < 0.01  (negligible class signal)
           KEEP        if 0.01 ≤ η² < 0.06  (small effect)
           STRONG_KEEP if η² ≥ 0.06  (medium or large effect)
    4. Save a ranked CSV and a color-coded horizontal bar chart.

    Outputs
    -------
    results/plots/feature_statistics/feature_selection_report.csv
    results/plots/feature_statistics/feature_selection_chart.png

    The CSV has columns: feature, group, sub_name, eta2, recommendation,
    effect_label — sorted by η² descending so the most discriminative features
    appear at the top.
    """
    logger.info("=" * 60)
    logger.info("Effect size feature selection  (split=%s)", split)
    logger.info("=" * 60)

    # ── 1. Compute η² across all groups ──────────────────────────────────────
    report_df = compute_effect_sizes(split)

    if report_df.empty:
        logger.error("No features could be loaded — run feature extraction first.")
        return

    # ── 2. Summary statistics ─────────────────────────────────────────────────
    counts = report_df["recommendation"].value_counts()
    total  = len(report_df)

    logger.info("\nη² summary across %d features:", total)
    for label in ("STRONG_KEEP", "KEEP", "DROP"):
        n = counts.get(label, 0)
        logger.info("  %-12s %3d  (%5.1f%%)", label, n, 100 * n / total)

    # Per-group breakdown
    logger.info("\nPer-group breakdown:")
    for group in INTERPRETABLE_GROUPS:
        g = report_df[report_df["group"] == group]
        if g.empty:
            continue
        drop_n  = (g["recommendation"] == "DROP").sum()
        keep_n  = (g["recommendation"] == "KEEP").sum()
        strong_n = (g["recommendation"] == "STRONG_KEEP").sum()
        logger.info(
            "  %-12s  STRONG_KEEP=%d  KEEP=%d  DROP=%d  (η²_max=%.4f)",
            group, strong_n, keep_n, drop_n, g["eta2"].max(),
        )

    # Top-10 most discriminative features
    logger.info("\nTop 10 most discriminative features:")
    for _, row in report_df.head(10).iterrows():
        logger.info(
            "  [%-12s] %-28s  η²=%.4f  (%s)",
            row["group"], row["feature"], row["eta2"], row["effect_label"],
        )

    # Features recommended for removal
    drop_df = report_df[report_df["recommendation"] == "DROP"]
    if not drop_df.empty:
        logger.info("\nFeatures recommended for DROP (η² < 0.01):")
        for _, row in drop_df.iterrows():
            logger.info(
                "  [%-12s] %-28s  η²=%.4f",
                row["group"], row["feature"], row["eta2"],
            )
    else:
        logger.info("\nNo features fall below the DROP threshold (η² < 0.01).")

    # ── 3. Save outputs ───────────────────────────────────────────────────────
    save_dir = PLOTS_DIR / "feature_statistics"
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "feature_selection_report.csv"
    report_df.to_csv(csv_path, index=False)
    logger.info("\nReport CSV → %s", csv_path)

    chart_path = save_dir / "feature_selection_chart.png"
    plot_effect_size_bar(report_df, chart_path)

    logger.info("\n%s", "=" * 60)
    logger.info("Done. Inspect the CSV and chart to decide which features to drop.")
    logger.info("Suggested threshold: remove features with η² < 0.01 (negligible).")
    logger.info("=" * 60)


# =============================================================================
# FEATURE MASK BUILDER  (for use by training pipeline)
# =============================================================================

# Ordered feature names within each branch tensor, matching GROUP_SUBFEATURES
# concatenation order in feature_loader.py.
#
# affective (34-dim): goemotions[28] + vad[3] + vader[3]
# handcrafted (26-dim): lexical[11] + syntactic[8] + structural[7]
#
# These lists MUST stay in sync with SUB_FEATURE_NAMES above.

_AFFECTIVE_FEATURE_ORDER: list[str] = (
    SUB_FEATURE_NAMES["affective"]["goemotions"]   # 28
    + SUB_FEATURE_NAMES["affective"]["vad"]        #  3
    + SUB_FEATURE_NAMES["affective"]["vader"]      #  3
)  # total: 34

_HANDCRAFTED_FEATURE_ORDER: list[str] = (
    SUB_FEATURE_NAMES["lexical"]["diversity"]      #  1
    + SUB_FEATURE_NAMES["lexical"]["word_rates"]   #  5
    + SUB_FEATURE_NAMES["lexical"]["pronouns"]     #  3
    + SUB_FEATURE_NAMES["lexical"]["punctuation"]  #  2  → lexical = 11
    + SUB_FEATURE_NAMES["syntactic"]["complexity"] #  3
    + SUB_FEATURE_NAMES["syntactic"]["pos_ratios"] #  3
    + SUB_FEATURE_NAMES["syntactic"]["readability"]#  2  → syntactic = 8
    + SUB_FEATURE_NAMES["structural"]["coherence"] #  4
    + SUB_FEATURE_NAMES["structural"]["tense"]     #  3  → structural = 7
)  # total: 26


def build_selection_masks(
    report_csv: "str | Path",
    threshold: str = "KEEP",
) -> dict[str, np.ndarray]:
    """
    Read a feature selection report CSV and build boolean keep-masks for the
    affective and handcrafted branches.

    Parameters
    ----------
    report_csv : path
        Path to ``feature_selection_report.csv`` produced by
        ``run_effect_size_selection()``.
    threshold : {"KEEP", "STRONG_KEEP"}
        Minimum recommendation to mark a feature as kept.
        "KEEP"        → keep everything with η² ≥ 0.01  (drops negligible only)
        "STRONG_KEEP" → keep only η² ≥ 0.06  (more aggressive)

    Returns
    -------
    dict with keys:
        "affective"    → np.ndarray bool, shape (34,)
        "handcrafted"  → np.ndarray bool, shape (26,)
        "n_kept"       → int, total features kept
        "n_dropped"    → int, total features dropped
        "dropped_names" → list[str], human-readable list of dropped features

    How masks are applied during training
    --------------------------------------
    ``feature_tensor *= mask``  zero-masks the dropped dimensions.
    Architecture is UNCHANGED — this is a zero-masking ablation, not
    dimension reduction.  The same model is trained with selected features
    forced to 0, making results directly comparable to the baseline.
    """
    import csv as _csv
    from pathlib import Path as _Path

    valid_thresholds = {"KEEP", "STRONG_KEEP"}
    if threshold not in valid_thresholds:
        raise ValueError(f"threshold must be one of {valid_thresholds}")

    _RANK = {"DROP": 0, "KEEP": 1, "STRONG_KEEP": 2}
    min_rank = _RANK[threshold]

    report_path = _Path(report_csv)
    if not report_path.exists():
        raise FileNotFoundError(
            f"Feature selection report not found: {report_path}\n"
            "Run: python -m scripts.main analyze-features --select"
        )

    # Read the CSV into a dict: feature_name → recommendation
    recs: dict[str, str] = {}
    with open(report_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            recs[row["feature"]] = row["recommendation"]

    def _make_mask(feature_order: list[str]) -> np.ndarray:
        mask = np.ones(len(feature_order), dtype=np.float32)
        for i, name in enumerate(feature_order):
            rec = recs.get(name, "KEEP")  # unknown features → keep by default
            if _RANK.get(rec, 1) < min_rank:
                mask[i] = 0.0
        return mask

    aff_mask = _make_mask(_AFFECTIVE_FEATURE_ORDER)
    hc_mask  = _make_mask(_HANDCRAFTED_FEATURE_ORDER)

    n_dropped = int((aff_mask == 0).sum() + (hc_mask == 0).sum())
    n_total   = len(aff_mask) + len(hc_mask)
    n_kept    = n_total - n_dropped

    dropped_names = (
        [n for n, m in zip(_AFFECTIVE_FEATURE_ORDER, aff_mask) if m == 0]
        + [n for n, m in zip(_HANDCRAFTED_FEATURE_ORDER, hc_mask) if m == 0]
    )

    logger.info(
        "Feature selection masks built  threshold=%s  "
        "kept=%d/%d  dropped=%d  (%s)",
        threshold, n_kept, n_total, n_dropped,
        ", ".join(dropped_names) if dropped_names else "none",
    )

    return {
        "affective":     aff_mask,
        "handcrafted":   hc_mask,
        "n_kept":        n_kept,
        "n_dropped":     n_dropped,
        "dropped_names": dropped_names,
    }


if __name__ == "__main__":
    import argparse

    from scripts.utils.logging_utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Feature statistics analysis (scaled heatmap or raw violin plots)"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw feature values and produce violin plots (no normalization)",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Run effect-size feature selection and export keep/drop report CSV",
    )
    args = parser.parse_args()

    if args.select:
        run_effect_size_selection()
    elif args.raw:
        run_feature_statistics_raw()
    else:
        run_feature_statistics()
