"""Tests for scripts/utils/outputs.py — path helper functions."""

import tempfile
import unittest
from pathlib import Path

import scripts.config as cfg
from scripts.utils.outputs import (
    checkpoint_dir,
    evaluation_dir,
    experiment_root,
    log_dir,
    training_dir,
)

_ORIG_RESULTS_DIR = cfg.RESULTS_DIR


class ExperimentRootTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg.RESULTS_DIR = Path(self._tmp.name)

    def tearDown(self):
        cfg.RESULTS_DIR = _ORIG_RESULTS_DIR
        self._tmp.cleanup()

    def test_fused_no_model_returns_fused_dir(self):
        root = experiment_root("fused")
        self.assertEqual(root.name, "fused")

    def test_fused_concat_returns_late_concat_subdir(self):
        root = experiment_root("fused", "concat")
        self.assertEqual(root.name, "late_concat")
        self.assertEqual(root.parent.name, "fused")

    def test_fused_gated_returns_gated_subdir(self):
        root = experiment_root("fused", "gated")
        self.assertEqual(root.name, "gated")
        self.assertEqual(root.parent.name, "fused")

    def test_single_group_no_model(self):
        root = experiment_root("semantic")
        self.assertEqual(root.name, "semantic")

    def test_single_group_with_model(self):
        root = experiment_root("semantic", "concat")
        self.assertEqual(root.name, "late_concat")
        self.assertEqual(root.parent.name, "semantic")

    def test_logistic_regression_name_mapping(self):
        root = experiment_root("fused", "logistic_regression")
        self.assertEqual(root.name, "logistic_regression")

    def test_unknown_model_uses_raw_name(self):
        root = experiment_root("fused", "my_custom_model")
        self.assertEqual(root.name, "my_custom_model")


class TrainingDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg.RESULTS_DIR = Path(self._tmp.name)

    def tearDown(self):
        cfg.RESULTS_DIR = _ORIG_RESULTS_DIR
        self._tmp.cleanup()

    def test_training_suffix(self):
        path = training_dir("fused", "concat")
        self.assertEqual(path.name, "training")

    def test_is_under_experiment_root(self):
        path = training_dir("semantic", "gated")
        self.assertEqual(path.parent, experiment_root("semantic", "gated"))


class EvaluationDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg.RESULTS_DIR = Path(self._tmp.name)

    def tearDown(self):
        cfg.RESULTS_DIR = _ORIG_RESULTS_DIR
        self._tmp.cleanup()

    def test_evaluation_suffix(self):
        path = evaluation_dir("fused", "gated")
        self.assertEqual(path.name, "evaluation")

    def test_is_under_experiment_root(self):
        path = evaluation_dir("fused", "gated")
        self.assertEqual(path.parent, experiment_root("fused", "gated"))


class CheckpointDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg.RESULTS_DIR = Path(self._tmp.name)

    def tearDown(self):
        cfg.RESULTS_DIR = _ORIG_RESULTS_DIR
        self._tmp.cleanup()

    def test_checkpoint_suffix(self):
        path = checkpoint_dir("fused", "concat")
        self.assertEqual(path.name, "checkpoints")

    def test_is_under_training_dir(self):
        path = checkpoint_dir("fused", "concat")
        self.assertEqual(path.parent, training_dir("fused", "concat"))


class LogDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg.RESULTS_DIR = Path(self._tmp.name)

    def tearDown(self):
        cfg.RESULTS_DIR = _ORIG_RESULTS_DIR
        self._tmp.cleanup()

    def test_log_suffix(self):
        path = log_dir("fused", "concat")
        self.assertEqual(path.name, "logs")

    def test_is_under_training_dir(self):
        path = log_dir("semantic", "gated")
        self.assertEqual(path.parent, training_dir("semantic", "gated"))

    def test_concat_and_gated_have_distinct_log_dirs(self):
        self.assertNotEqual(log_dir("fused", "concat"), log_dir("fused", "gated"))

    def test_fused_and_single_group_have_distinct_log_dirs(self):
        self.assertNotEqual(log_dir("fused", "concat"), log_dir("semantic", "concat"))


if __name__ == "__main__":
    unittest.main()
