"""
Tests for scripts/models/fusion/feature_loader.py.

Group directories are temporarily replaced with temp dirs containing minimal
parquet files so no real data on disk is required.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import scripts.data.feature_loader as loader_mod
from scripts import config
from scripts.data.feature_loader import (
    _normalize_post_ids,
    _subextractor_feature_path,
    load_flat_feature_matrix,
    load_group_features,
    load_subextractor_features,
)

_ORIG_GROUP_DIRS = dict(loader_mod.GROUP_DIRS)


def _write_feature_parquet(
    directory: Path, name: str, n_rows: int, dim: int, split: str | None = None
) -> Path:
    """Write a minimal feature parquet to a directory (optionally under a split sub-dir)."""
    target_dir = directory / split if split else directory
    target_dir.mkdir(parents=True, exist_ok=True)
    post_ids = [f"{split or 'test'}_{i}" for i in range(n_rows)]
    features = [np.ones(dim, dtype=float).tolist() for _ in range(n_rows)]
    path = target_dir / f"{name}.parquet"
    pd.DataFrame({"post_id": post_ids, "features": features}).to_parquet(path, index=False)
    return path


class NormalizePostIdsTests(unittest.TestCase):
    def test_strips_split_prefix_when_suffix_is_digit(self):
        result = _normalize_post_ids(["train_0", "train_1", "train_42"])
        self.assertEqual(result, ["0", "1", "42"])

    def test_strips_val_prefix(self):
        result = _normalize_post_ids(["val_5", "val_10"])
        self.assertEqual(result, ["5", "10"])

    def test_leaves_plain_ids_unchanged(self):
        result = _normalize_post_ids(["abc", "xyz"])
        self.assertEqual(result, ["abc", "xyz"])

    def test_leaves_ids_without_digit_suffix_unchanged(self):
        result = _normalize_post_ids(["train_foo", "prefix_bar"])
        self.assertEqual(result, ["train_foo", "prefix_bar"])

    def test_empty_list(self):
        self.assertEqual(_normalize_post_ids([]), [])

    def test_mixed_input(self):
        result = _normalize_post_ids(["train_3", "plain", "val_7"])
        self.assertEqual(result, ["3", "plain", "7"])


class SubextractorFeaturePathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        loader_mod.GROUP_DIRS = {
            "semantic": self._base / "semantic",
            "lexical": self._base / "lexical",
        }

    def tearDown(self):
        loader_mod.GROUP_DIRS = _ORIG_GROUP_DIRS
        self._tmp.cleanup()

    def test_returns_base_path_when_no_split(self):
        path = _subextractor_feature_path("semantic", "mental_roberta")
        self.assertEqual(path.name, "mental_roberta.parquet")
        self.assertEqual(path.parent.name, "semantic")

    def test_returns_split_subdir_path_when_split_given(self):
        path = _subextractor_feature_path("lexical", "diversity", split="train")
        self.assertEqual(path.name, "diversity.parquet")
        self.assertEqual(path.parent.name, "train")


class LoadSubextractorFeaturesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        loader_mod.GROUP_DIRS = {
            "semantic": self._base / "semantic",
            "lexical": self._base / "lexical",
        }

    def tearDown(self):
        loader_mod.GROUP_DIRS = _ORIG_GROUP_DIRS
        self._tmp.cleanup()

    def test_loads_correct_shape(self):
        _write_feature_parquet(self._base / "semantic", "mental_roberta", n_rows=5, dim=768)
        ids, matrix = load_subextractor_features("semantic", "mental_roberta")
        self.assertEqual(matrix.shape, (5, 768))
        self.assertEqual(len(ids), 5)

    def test_returns_float32(self):
        _write_feature_parquet(self._base / "semantic", "mental_roberta", n_rows=3, dim=4)
        _, matrix = load_subextractor_features("semantic", "mental_roberta")
        self.assertEqual(matrix.dtype, np.float32)

    def test_split_aware_load(self):
        _write_feature_parquet(
            self._base / "lexical", "diversity", n_rows=4, dim=1, split="train"
        )
        ids, matrix = load_subextractor_features("lexical", "diversity", split="train")
        self.assertEqual(matrix.shape, (4, 1))

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_subextractor_features("semantic", "mental_roberta", split="train")


class LoadGroupFeaturesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        loader_mod.GROUP_DIRS = {g: self._base / g for g in loader_mod.GROUP_SUBFEATURES}

    def tearDown(self):
        loader_mod.GROUP_DIRS = _ORIG_GROUP_DIRS
        self._tmp.cleanup()

    def _write_group(self, group: str, split: str, n_rows: int) -> None:
        for sub_name in loader_mod.GROUP_SUBFEATURES[group]:
            dim = {"mental_roberta": 768, "diversity": 1, "word_rates": 5,
                   "pronouns": 3, "punctuation": 2, "complexity": 3, "pos_ratios": 3,
                   "readability": 2, "coherence": 4, "tense": 3,
                   "goemotions": 28, "vad": 3, "vader": 3}.get(sub_name, 4)
            _write_feature_parquet(
                self._base / group, sub_name, n_rows=n_rows, dim=dim, split=split
            )

    def test_semantic_concatenated_shape(self):
        self._write_group("semantic", "train", n_rows=6)
        ids, matrix = load_group_features("semantic", split="train")
        self.assertEqual(matrix.shape, (6, 768))
        self.assertEqual(len(ids), 6)

    def test_affective_concatenated_shape(self):
        # goemotions(28) + vad(3) + vader(3) = 34
        self._write_group("affective", "train", n_rows=4)
        _, matrix = load_group_features("affective", split="train")
        self.assertEqual(matrix.shape[1], 34)

    def test_lexical_concatenated_shape(self):
        # diversity(1) + word_rates(5) + pronouns(3) + punctuation(2) = 11
        self._write_group("lexical", "val", n_rows=3)
        _, matrix = load_group_features("lexical", split="val")
        self.assertEqual(matrix.shape[1], 11)

    def test_post_id_mismatch_raises(self):
        # Write one sub-extractor with different row counts to trigger mismatch
        _write_feature_parquet(
            self._base / "lexical", "diversity", n_rows=3, dim=1, split="train"
        )
        _write_feature_parquet(
            self._base / "lexical", "word_rates", n_rows=4, dim=5, split="train"
        )
        _write_feature_parquet(
            self._base / "lexical", "pronouns", n_rows=4, dim=3, split="train"
        )
        _write_feature_parquet(
            self._base / "lexical", "punctuation", n_rows=4, dim=2, split="train"
        )
        with self.assertRaises(AssertionError):
            load_group_features("lexical", split="train")

    def test_unknown_group_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_group_features("nonexistent", split="train")


class LoadFeatureTensorsTests(unittest.TestCase):
    """Test load_feature_tensors for single-group (ablation) mode."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        loader_mod.GROUP_DIRS = {g: self._base / g for g in loader_mod.GROUP_SUBFEATURES}

    def tearDown(self):
        loader_mod.GROUP_DIRS = _ORIG_GROUP_DIRS
        self._tmp.cleanup()

    def _write_semantic(self, n_rows: int = 4) -> None:
        _write_feature_parquet(
            self._base / "semantic", "mental_roberta", n_rows=n_rows, dim=768, split="train"
        )

    def test_semantic_single_group_fills_semantic_branch(self):
        self._write_semantic(n_rows=4)
        from scripts.data.feature_loader import load_feature_tensors
        sem, aff, hc, ids = load_feature_tensors("semantic", split="train")
        self.assertEqual(sem.shape, (4, config.SEMANTIC_DIM))
        # Affective and handcrafted branches should be zero-filled
        self.assertTrue((aff == 0).all())
        self.assertTrue((hc == 0).all())

    def test_affective_single_group_fills_affective_branch(self):
        for sub_name, dim in [("goemotions", 28), ("vad", 3), ("vader", 3)]:
            _write_feature_parquet(
                self._base / "affective", sub_name, n_rows=5, dim=dim, split="train"
            )
        from scripts.data.feature_loader import load_feature_tensors
        sem, aff, hc, ids = load_feature_tensors("affective", split="train")
        self.assertEqual(aff.shape, (5, config.AFFECTIVE_DIM))
        self.assertTrue((sem == 0).all())
        self.assertTrue((hc == 0).all())

    def test_tensors_are_torch_float32(self):
        self._write_semantic(n_rows=3)
        from scripts.data.feature_loader import load_feature_tensors
        sem, aff, hc, _ = load_feature_tensors("semantic", split="train")
        for t in (sem, aff, hc):
            self.assertIsInstance(t, torch.Tensor)
            self.assertEqual(t.dtype, torch.float32)

    def test_invalid_config_raises_value_error(self):
        from scripts.data.feature_loader import load_feature_tensors
        with self.assertRaises(ValueError):
            load_feature_tensors("invalid_group", split="train")


class LoadFlatFeatureMatrixTests(unittest.TestCase):
    """Test load_flat_feature_matrix for single-group classical classifiers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        loader_mod.GROUP_DIRS = {g: self._base / g for g in loader_mod.GROUP_SUBFEATURES}

    def tearDown(self):
        loader_mod.GROUP_DIRS = _ORIG_GROUP_DIRS
        self._tmp.cleanup()

    def test_single_group_returns_correct_matrix(self):
        _write_feature_parquet(
            self._base / "semantic", "mental_roberta", n_rows=6, dim=768, split="val"
        )
        ids, matrix = load_flat_feature_matrix("semantic", split="val")
        self.assertEqual(matrix.shape, (6, 768))
        self.assertEqual(len(ids), 6)

    def test_returns_float32(self):
        _write_feature_parquet(
            self._base / "semantic", "mental_roberta", n_rows=3, dim=4, split="test"
        )
        _, matrix = load_flat_feature_matrix("semantic", split="test")
        self.assertEqual(matrix.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
