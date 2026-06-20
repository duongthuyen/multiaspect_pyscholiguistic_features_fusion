"""
Tests for scripts/utils/combined_features_info.py.

All I/O is isolated to temporary directories; no shared state is modified.
Config paths are temporarily replaced and always restored in tearDown.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.config as cfg
import scripts.analysis.combined_features_info as mod
from scripts.analysis.combined_features_info import (
    export_combined_summary,
    get_combined_info,
    load_all_combined_tensors,
    load_combined_features,
    validate_combined_consistency,
)

_ORIG_FEATURES_DIR = cfg.FEATURES_DIR
_ORIG_GROUPS = list(mod.GROUPS)
_ORIG_SPLITS = list(mod.SPLITS)


def _write_parquet(base: Path, group: str, split: str, n_rows: int = 4, dim: int = 8) -> None:
    out_dir = base / group / split
    out_dir.mkdir(parents=True, exist_ok=True)
    post_ids = [f"{split}_{i}" for i in range(n_rows)]
    features = [np.ones(dim, dtype=float).tolist() for _ in range(n_rows)]
    pd.DataFrame({"post_id": post_ids, "features": features}).to_parquet(
        out_dir / "combined.parquet", index=False
    )


class LoadCombinedFeaturesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        _write_parquet(self._base, "semantic", "train", n_rows=5, dim=768)
        cfg.FEATURES_DIR = self._base

    def tearDown(self):
        cfg.FEATURES_DIR = _ORIG_FEATURES_DIR
        self._tmp.cleanup()

    def test_returns_dataframe_with_expected_columns(self):
        df = load_combined_features("semantic", "train")
        self.assertIn("post_id", df.columns)
        self.assertIn("features", df.columns)

    def test_correct_row_count(self):
        df = load_combined_features("semantic", "train")
        self.assertEqual(len(df), 5)

    def test_feature_dimension(self):
        df = load_combined_features("semantic", "train")
        self.assertEqual(len(df["features"].iloc[0]), 768)

    def test_post_ids_are_strings(self):
        df = load_combined_features("semantic", "train")
        self.assertTrue(all(isinstance(pid, str) for pid in df["post_id"]))

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_combined_features("lexical", "train")


class GetCombinedInfoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        _write_parquet(self._base, "semantic", "val", n_rows=3, dim=768)
        _write_parquet(self._base, "lexical", "val", n_rows=3, dim=11)
        cfg.FEATURES_DIR = self._base

    def tearDown(self):
        cfg.FEATURES_DIR = _ORIG_FEATURES_DIR
        self._tmp.cleanup()

    def test_success_contains_rows_and_dim(self):
        info = get_combined_info("semantic", "val")
        self.assertNotIn("error", info)
        self.assertEqual(info["rows"], 3)
        self.assertEqual(info["feature_dim"], 768)

    def test_success_sample_post_ids_present(self):
        info = get_combined_info("lexical", "val")
        self.assertIn("sample_post_ids", info)
        self.assertLessEqual(len(info["sample_post_ids"]), 3)

    def test_missing_group_returns_error_key(self):
        info = get_combined_info("nonexistent", "val")
        self.assertIn("error", info)
        self.assertNotIn("rows", info)


class LoadAllCombinedTensorsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        _write_parquet(self._base, "semantic", "train", n_rows=4, dim=8)
        _write_parquet(self._base, "lexical", "train", n_rows=4, dim=3)
        cfg.FEATURES_DIR = self._base
        mod.GROUPS = ["semantic", "lexical"]
        mod.SPLITS = ["train"]

    def tearDown(self):
        cfg.FEATURES_DIR = _ORIG_FEATURES_DIR
        mod.GROUPS = _ORIG_GROUPS
        mod.SPLITS = _ORIG_SPLITS
        self._tmp.cleanup()

    def test_present_groups_loaded_as_float32_arrays(self):
        tensors = load_all_combined_tensors("train")
        self.assertIn("semantic", tensors)
        self.assertIn("lexical", tensors)
        self.assertIsInstance(tensors["semantic"]["features"], np.ndarray)
        self.assertEqual(tensors["semantic"]["features"].dtype, np.float32)

    def test_shapes_match_written_data(self):
        tensors = load_all_combined_tensors("train")
        self.assertEqual(tensors["semantic"]["features"].shape, (4, 8))
        self.assertEqual(tensors["lexical"]["features"].shape, (4, 3))

    def test_post_ids_are_lists(self):
        tensors = load_all_combined_tensors("train")
        self.assertIsInstance(tensors["semantic"]["post_ids"], list)

    def test_missing_group_is_absent_from_result(self):
        mod.GROUPS = ["semantic", "missing_group"]
        tensors = load_all_combined_tensors("train")
        self.assertIn("semantic", tensors)
        self.assertNotIn("missing_group", tensors)


class ValidateCombinedConsistencyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        cfg.FEATURES_DIR = self._base
        mod.SPLITS = ["train"]

    def tearDown(self):
        cfg.FEATURES_DIR = _ORIG_FEATURES_DIR
        mod.GROUPS = _ORIG_GROUPS
        mod.SPLITS = _ORIG_SPLITS
        self._tmp.cleanup()

    def test_consistent_post_ids_returns_true(self):
        _write_parquet(self._base, "semantic", "train", n_rows=3, dim=4)
        _write_parquet(self._base, "lexical", "train", n_rows=3, dim=4)
        mod.GROUPS = ["semantic", "lexical"]
        self.assertTrue(validate_combined_consistency())

    def test_different_row_counts_returns_false(self):
        _write_parquet(self._base, "semantic", "train", n_rows=3, dim=4)
        _write_parquet(self._base, "lexical", "train", n_rows=5, dim=4)
        mod.GROUPS = ["semantic", "lexical"]
        self.assertFalse(validate_combined_consistency())

    def test_missing_group_file_returns_false(self):
        _write_parquet(self._base, "semantic", "train", n_rows=3, dim=4)
        mod.GROUPS = ["semantic", "nonexistent"]
        self.assertFalse(validate_combined_consistency())

    def test_single_group_trivially_consistent(self):
        _write_parquet(self._base, "semantic", "train", n_rows=3, dim=4)
        mod.GROUPS = ["semantic"]
        self.assertTrue(validate_combined_consistency())


class ExportCombinedSummaryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        _write_parquet(self._base, "semantic", "train", n_rows=2, dim=4)
        cfg.FEATURES_DIR = self._base
        mod.GROUPS = ["semantic"]
        mod.SPLITS = ["train"]

    def tearDown(self):
        cfg.FEATURES_DIR = _ORIG_FEATURES_DIR
        mod.GROUPS = _ORIG_GROUPS
        mod.SPLITS = _ORIG_SPLITS
        self._tmp.cleanup()

    def test_creates_json_file(self):
        export_combined_summary()
        self.assertTrue((self._base / "combined_summary.json").exists())

    def test_json_has_expected_structure(self):
        export_combined_summary()
        with open(self._base / "combined_summary.json") as f:
            data = json.load(f)
        self.assertIn("train", data)
        self.assertIn("semantic", data["train"])
        self.assertEqual(data["train"]["semantic"]["rows"], 2)

    def test_returns_summary_dict(self):
        summary = export_combined_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn("train", summary)


if __name__ == "__main__":
    unittest.main()
