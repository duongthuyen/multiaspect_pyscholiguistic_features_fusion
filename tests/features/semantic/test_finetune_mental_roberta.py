"""
Tests for scripts/features/semantic/finetune_mental_roberta.py.

The actual fine-tuning (which downloads a 500MB model) is not run.
Only the dataset helper, the seed function, and the extraction-and-save
path are tested using fake objects and temporary directories.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import scripts.config as cfg
from scripts.features.semantic.finetune_mental_roberta import (
    MentalHealthDataset,
    extract_and_save_cls,
    set_seed,
)

_ORIG_TRAIN_PATH = cfg.TRAIN_PATH
_ORIG_VAL_PATH = cfg.VAL_PATH
_ORIG_TEST_PATH = cfg.TEST_PATH
_ORIG_SEMANTIC_DIR = cfg.SEMANTIC_FEATURES_DIR


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Returns fixed-length token tensors without calling any real model."""

    def __init__(self, max_length: int = 8):
        self._max_length = max_length

    def __call__(self, text: str, max_length: int, padding: str, truncation: bool, return_tensors: str):
        length = min(max_length, self._max_length)
        return {
            "input_ids": torch.ones((1, length), dtype=torch.long),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }


class _FakeExtractor:
    """Returns a fixed numpy matrix without loading any transformer model."""

    def __init__(self, dim: int = 768, fill: float = 1.0):
        self._dim = dim
        self._fill = fill

    def extract_batch(self, texts: list[str]) -> np.ndarray:
        return np.full((len(texts), self._dim), self._fill, dtype=np.float32)


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------


class SetSeedTests(unittest.TestCase):
    def test_torch_determinism(self):
        set_seed(99)
        a = torch.randn(5, 5)
        set_seed(99)
        b = torch.randn(5, 5)
        self.assertTrue(torch.equal(a, b))

    def test_numpy_determinism(self):
        set_seed(7)
        a = np.random.rand(3)
        set_seed(7)
        b = np.random.rand(3)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(1)
        a = torch.randn(10)
        set_seed(2)
        b = torch.randn(10)
        self.assertFalse(torch.equal(a, b))


# ---------------------------------------------------------------------------
# MentalHealthDataset
# ---------------------------------------------------------------------------


def _make_df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            cfg.TEXT_COL: [f"sample text number {i}" for i in range(n)],
            cfg.LABEL_COL: [i % cfg.NUM_LABELS for i in range(n)],
        }
    ).iloc[:n]


class MentalHealthDatasetTests(unittest.TestCase):
    def _make(self, n: int = 4, max_length: int = 8) -> MentalHealthDataset:
        return MentalHealthDataset(_make_df(n), _FakeTokenizer(max_length), max_length)

    def test_length_matches_dataframe(self):
        ds = self._make(n=6)
        self.assertEqual(len(ds), 6)

    def test_item_has_required_keys(self):
        ds = self._make()
        item = ds[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)

    def test_input_ids_shape(self):
        ds = self._make(max_length=8)
        item = ds[0]
        self.assertEqual(item["input_ids"].shape, (8,))

    def test_attention_mask_shape_matches_input_ids(self):
        ds = self._make(max_length=6)
        item = ds[0]
        self.assertEqual(item["attention_mask"].shape, item["input_ids"].shape)

    def test_labels_is_long_tensor(self):
        ds = self._make()
        item = ds[0]
        self.assertEqual(item["labels"].dtype, torch.long)
        self.assertEqual(item["labels"].ndim, 0)  # scalar

    def test_label_value_within_range(self):
        ds = self._make(n=cfg.NUM_LABELS)
        for i in range(len(ds)):
            label = int(ds[i]["labels"].item())
            self.assertGreaterEqual(label, 0)
            self.assertLess(label, cfg.NUM_LABELS)

    def test_nan_text_does_not_crash(self):
        df = _make_df(n=2)
        df.iloc[0, df.columns.get_loc(cfg.TEXT_COL)] = float("nan")
        ds = MentalHealthDataset(df, _FakeTokenizer(), max_length=8)
        item = ds[0]
        self.assertIn("input_ids", item)

    def test_single_row_dataset(self):
        ds = self._make(n=1)
        self.assertEqual(len(ds), 1)
        item = ds[0]
        self.assertIn("labels", item)


# ---------------------------------------------------------------------------
# extract_and_save_cls
# ---------------------------------------------------------------------------


class ExtractAndSaveClsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)

        # Write minimal CSV files for each split
        for split in ("train", "val", "test"):
            df = _make_df(n=4)
            path = base / f"{split}.csv"
            df.to_csv(path, index=False)
            setattr(cfg, f"{split.upper()}_PATH", path)

        cfg.SEMANTIC_FEATURES_DIR = base / "features" / "semantic"

    def tearDown(self):
        cfg.TRAIN_PATH = _ORIG_TRAIN_PATH
        cfg.VAL_PATH = _ORIG_VAL_PATH
        cfg.TEST_PATH = _ORIG_TEST_PATH
        cfg.SEMANTIC_FEATURES_DIR = _ORIG_SEMANTIC_DIR
        self._tmp.cleanup()

    def _run(self, split: str = "train", dim: int = 768, n_rows: int = 4) -> Path:
        extractor = _FakeExtractor(dim=dim, fill=0.5)
        extract_and_save_cls(split, extractor)
        return cfg.SEMANTIC_FEATURES_DIR / split / "mental_roberta.parquet"

    def test_creates_parquet_file(self):
        path = self._run()
        self.assertTrue(path.exists(), msg=f"Expected parquet at {path}")

    def test_parquet_has_correct_columns(self):
        path = self._run()
        df = pd.read_parquet(path)
        self.assertIn("post_id", df.columns)
        self.assertIn("features", df.columns)

    def test_parquet_row_count_matches_split(self):
        path = self._run()
        df = pd.read_parquet(path)
        self.assertEqual(len(df), 4)

    def test_post_ids_have_correct_prefix(self):
        path = self._run(split="train")
        df = pd.read_parquet(path)
        for pid in df["post_id"]:
            self.assertTrue(str(pid).startswith("train_"), msg=f"Unexpected post_id: {pid}")

    def test_feature_dimension_matches_extractor(self):
        path = self._run(dim=768)
        df = pd.read_parquet(path)
        self.assertEqual(len(df["features"].iloc[0]), 768)

    def test_feature_values_are_finite(self):
        path = self._run()
        df = pd.read_parquet(path)
        for features in df["features"]:
            self.assertTrue(all(np.isfinite(v) for v in features))

    def test_val_split_writes_to_val_subdir(self):
        path = self._run(split="val")
        self.assertEqual(path.parent.name, "val")

    def test_test_split_writes_to_test_subdir(self):
        path = self._run(split="test")
        self.assertEqual(path.parent.name, "test")

    def test_overwrites_existing_parquet(self):
        # Write once with fill=0.5, then again with fill=0.9; check second value
        self._run(split="train")
        extractor2 = _FakeExtractor(dim=768, fill=0.9)
        extract_and_save_cls("train", extractor2)
        df = pd.read_parquet(cfg.SEMANTIC_FEATURES_DIR / "train" / "mental_roberta.parquet")
        first_val = df["features"].iloc[0][0]
        self.assertAlmostEqual(first_val, 0.9, places=4)


if __name__ == "__main__":
    unittest.main()
