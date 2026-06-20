# scripts/data/analysis.py

import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scripts.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,  # Processed CSVs to read from
    TEXT_COL, LABEL_COL, CLASS_NAME_COL,
    PLOTS_DIR,                         # Where to save figures
    MAX_LENGTH                         # RoBERTa token limit (512)
)

logger = logging.getLogger(__name__)


# =============================================================================
# PLOT STYLE
# =============================================================================
# Setting a consistent style here means every plot in this file looks uniform.
# "whitegrid" gives clean plots suitable for a thesis or paper.
# The palette is colorblind-friendly.

sns.set_theme(style="whitegrid")
PALETTE = "Set2"


# =============================================================================
# STEP 1 — LOAD PROCESSED DATA
# =============================================================================

def load_processed_splits() -> dict[str, pd.DataFrame]:
    """
    Load the three processed CSV splits.

    Input:  None — paths from config.py
    Output: dict {"train": df, "val": df, "test": df}

    Note: we load from PROCESSED paths here, not RAW.
    Analysis always runs on clean data, never raw.
    """
    splits = {
        "train" : TRAIN_PATH,
        "val"   : VAL_PATH,
        "test"  : TEST_PATH,
    }

    datasets = {}
    for name, path in splits.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Processed file not found: {path}\n"
                f"Run preprocessing.py first."
            )

        df = pd.read_csv(path)
        datasets[name] = df
        logger.info(f"Loaded {name}: {df.shape[0]:,} rows")

    return datasets


# =============================================================================
# STEP 2 — COMPUTE TEXT LENGTH FEATURES
# =============================================================================
# We add three length columns to each DataFrame.
# These are not saved — they are only used for analysis within this script.

def count_words(text: str) -> int:
    """Count words by splitting on whitespace."""
    return len(str(text).split())

def count_sentences(text: str) -> int:
    """
    Count sentences by splitting on sentence-ending punctuation.
    We use a simple rule here because this is only for analysis,
    not for feature extraction. Speed matters more than perfection.
    """
    sentences = re.split(r'[.!?]+', str(text))
    # Filter out empty strings that result from the split
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

def add_length_columns(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Add word_count and sentence_count columns to each split.

    Input:  datasets dict (processed)
    Output: datasets dict with two new columns added per DataFrame

    Why not token count? True tokenization (like RoBERTa's BPE) requires
    loading the tokenizer which is slow. Word count is a good enough proxy
    for EDA purposes and runs instantly.
    """
    for name, df in datasets.items():
        df["word_count"]     = df[TEXT_COL].apply(count_words)
        df["sentence_count"] = df[TEXT_COL].apply(count_sentences)
        logger.info(f"{name} — added word_count and sentence_count columns")

    return datasets


# =============================================================================
# STEP 3 — CLASS DISTRIBUTION
# =============================================================================

def print_class_distribution(datasets: dict[str, pd.DataFrame]) -> None:
    """
    Print the number of samples per class for each split.

    Input:  datasets dict
    Output: None — prints to console and logger

    This answers: is the dataset balanced?
    For your thesis: report these numbers in the Dataset section.
    """
    for name, df in datasets.items():
        logger.info(f"\n--- Class distribution: {name} ---")
        dist = df[CLASS_NAME_COL].value_counts().sort_index()
        for class_name, count in dist.items():
            logger.info(f"  {class_name:<20} {count:>6,} samples")


def plot_class_distribution(datasets: dict[str, pd.DataFrame], save_dir: Path) -> None:
    """
    Save a bar chart of class distribution for each split.

    Input:  datasets dict, save_dir Path to save figures
    Output: None — saves PNG files to results/plots/analysis/

    Why one plot per split? Train/val/test should have similar distributions
    due to stratified splitting. Plotting all three lets you visually verify
    this is the case — if they look very different, something went wrong.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, df in datasets.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        # value_counts() counts occurrences of each class.
        # sort_index() sorts alphabetically so the order is consistent
        # across all three plots.
        counts = df[CLASS_NAME_COL].value_counts().sort_index()

        sns.barplot(
            x=counts.index,
            y=counts.values,
            palette=PALETTE,
            ax=ax
        )

        ax.set_title(f"Class Distribution — {name} set", fontsize=14, pad=25)
        ax.set_xlabel("Mental Health Condition", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)

        # Use the source data directly — get_xticklabels() returns empty Text
        # objects before plt.draw() in seaborn 0.13+, causing silent label loss.
        ax.set_xticklabels(
            [str(label).capitalize() for label in counts.index], rotation=30
        )

        # Add count labels on top of each bar for readability
        for bar, count in zip(ax.patches, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{count:,}",
                ha="center", va="bottom", fontsize=10
            )

        plt.tight_layout()
        save_path = save_dir / f"class_distribution_{name}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved class distribution plot → {save_path}")


# =============================================================================
# STEP 4 — TEXT LENGTH DISTRIBUTION
# =============================================================================

def print_length_statistics(datasets: dict[str, pd.DataFrame]) -> None:
    """
    Print summary statistics for word count and sentence count.

    Input:  datasets dict (with length columns added)
    Output: None — prints to console

    For your thesis: report mean, median, min, max word count.
    The difference between mean and median tells you about skew —
    if mean >> median, a small number of very long posts are pulling
    the average up.
    """
    for name, df in datasets.items():
        logger.info(f"\n--- Length statistics: {name} ---")
        for col in ["word_count", "sentence_count"]:
            stats = df[col].describe()
            logger.info(
                f"  {col}: "
                f"mean={stats['mean']:.1f}, "
                f"median={df[col].median():.1f}, "
                f"min={stats['min']:.0f}, "
                f"max={stats['max']:.0f}"
            )


def plot_length_distribution(datasets: dict[str, pd.DataFrame], save_dir: Path) -> None:
    """
    Save word count distribution plots, separated by class.

    Input:  datasets dict (with word_count column), save_dir
    Output: None — saves PNG files

    Why separate by class? A single distribution hides the fact that
    different conditions may have very different writing lengths.
    Plotting per class reveals this — Depression posts tend to be longer
    due to more elaborate self-expression. This finding motivates your
    structural features.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # We only plot the training set here — it has the most samples
    # and is the most representative for reporting in your thesis.
    df = datasets["train"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot one distribution curve per class using KDE (Kernel Density Estimate).
    # KDE draws a smooth curve instead of a histogram, which is cleaner
    # when comparing multiple groups on the same axes.
    classes = sorted(df[CLASS_NAME_COL].unique())
    for cls in classes:
        subset = df[df[CLASS_NAME_COL] == cls]["word_count"]
        sns.kdeplot(subset, label=cls, ax=ax, fill=False)

    # Add a vertical line at RoBERTa's token limit (512).
    # This visually shows what proportion of posts get truncated.
    ax.axvline(
        x=MAX_LENGTH,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"RoBERTa limit ({MAX_LENGTH} tokens)"
    )

    ax.set_title("Word Count Distribution by Class — train set", fontsize=14)
    ax.set_xlabel("Word Count", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1500)   # Cap x-axis to avoid extreme outliers distorting the plot

    plt.tight_layout()
    save_path = save_dir / "word_count_by_class.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved length distribution plot → {save_path}")


# =============================================================================
# STEP 5 — MISSING CONTENT ANALYSIS
# =============================================================================

def print_empty_post_analysis(datasets: dict[str, pd.DataFrame]) -> None:
    """
    Report how many posts have very short or empty text after merging.

    Input:  datasets dict (with word_count column)
    Output: None — prints to console

    Why does this matter? A post with word_count <= 3 is likely just
    a title with no body (e.g. "I feel hopeless: "). These posts
    carry very little linguistic signal. Knowing how many exist
    tells you whether you need a minimum length filter.
    We report but do not remove them — removal is a preprocessing
    decision that must be explicitly justified in your thesis.
    """
    for name, df in datasets.items():
        very_short = (df["word_count"] <= 3).sum()
        total = len(df)
        pct = very_short / total * 100
        logger.info(
            f"{name} — posts with <= 3 words: "
            f"{very_short} ({pct:.1f}% of {total:,})"
        )


# =============================================================================
# STEP 6 — PER-CLASS LENGTH SUMMARY TABLE
# =============================================================================

def print_per_class_length_table(datasets: dict[str, pd.DataFrame]) -> None:
    """
    Print a table of mean word count per class for the training set.

    Input:  datasets dict (with word_count column)
    Output: None — prints formatted table to console

    This is directly reportable in your thesis as a summary table.
    Differences in mean length across classes motivate structural
    and syntactic features — conditions with longer posts likely
    show more complex sentence organization.
    """
    df = datasets["train"]

    table = df.groupby(CLASS_NAME_COL)["word_count"].agg(
        Mean="mean",
        Median="median",
        Min="min",
        Max="max"
    ).round(1)

    logger.info("\n--- Per-class word count (train set) ---")
    logger.info("\n" + table.to_string())


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def analyze() -> None:
    """
    Run the full EDA pipeline in order.
    Called by main.py with: python -m scripts.main --stage analyze
    """
    save_dir = PLOTS_DIR / "EDA"

    logger.info("=" * 60)
    logger.info("Starting data analysis")
    logger.info("=" * 60)

    datasets = load_processed_splits()
    datasets = add_length_columns(datasets)

    print_class_distribution(datasets)
    plot_class_distribution(datasets, save_dir)

    print_length_statistics(datasets)
    plot_length_distribution(datasets, save_dir)

    print_empty_post_analysis(datasets)
    print_per_class_length_table(datasets)

    logger.info("=" * 60)
    logger.info("Analysis complete. Plots saved to results/plots/EDA/")
    logger.info("=" * 60)


if __name__ == "__main__":
    analyze()