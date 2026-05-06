"""Tests for scripts/evaluation/metrics.py — save_confusion_matrix_artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluation.metrics import save_confusion_matrix_artifacts

_CLASS_NAMES = ["A", "B", "C"]
_Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
_Y_PRED = np.array([0, 1, 1, 1, 2, 0])


class SaveConfusionMatrixArtifactsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._out = Path(self._tmp.name) / "cm"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, y_true=_Y_TRUE, y_pred=_Y_PRED, class_names=_CLASS_NAMES):
        return save_confusion_matrix_artifacts(y_true, y_pred, class_names, self._out)

    def test_returns_dict_with_expected_keys(self):
        result = self._run()
        for key in ("confusion_matrix", "raw_csv_path", "raw_json_path", "plot_path"):
            self.assertIn(key, result)

    def test_csv_file_created(self):
        result = self._run()
        self.assertTrue(Path(result["raw_csv_path"]).exists())

    def test_json_file_created(self):
        result = self._run()
        self.assertTrue(Path(result["raw_json_path"]).exists())

    def test_png_file_created(self):
        result = self._run()
        self.assertTrue(Path(result["plot_path"]).exists())

    def test_json_contains_correct_class_names(self):
        result = self._run()
        with open(result["raw_json_path"]) as f:
            data = json.load(f)
        self.assertEqual(data["class_names"], _CLASS_NAMES)

    def test_confusion_matrix_shape(self):
        result = self._run()
        cm = result["confusion_matrix"]
        self.assertEqual(len(cm), len(_CLASS_NAMES))
        self.assertEqual(len(cm[0]), len(_CLASS_NAMES))

    def test_confusion_matrix_diagonal_values(self):
        # y_true=[0,0,1,1,2,2], y_pred=[0,1,1,1,2,0]
        # Row 0: true 0 predicted as 0 once, 1 once → cm[0][0]=1, cm[0][1]=1
        # Row 1: true 1 predicted as 1 twice → cm[1][1]=2
        # Row 2: true 2 predicted as 2 once, 0 once → cm[2][2]=1, cm[2][0]=1
        result = self._run()
        cm = result["confusion_matrix"]
        self.assertEqual(cm[0][0], 1)
        self.assertEqual(cm[1][1], 2)
        self.assertEqual(cm[2][2], 1)

    def test_csv_row_count_matches_classes(self):
        self._run()
        csv_path = self._out / "confusion_matrix.csv"
        with open(csv_path) as f:
            rows = [line for line in f if line.strip()]
        self.assertEqual(len(rows), len(_CLASS_NAMES))

    def test_output_directory_created_automatically(self):
        nested = Path(self._tmp.name) / "deep" / "nested" / "cm"
        save_confusion_matrix_artifacts(_Y_TRUE, _Y_PRED, _CLASS_NAMES, nested)
        self.assertTrue(nested.exists())

    def test_perfect_prediction_all_diagonal(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        result = save_confusion_matrix_artifacts(y_true, y_pred, _CLASS_NAMES, self._out)
        cm = result["confusion_matrix"]
        for i in range(3):
            self.assertEqual(cm[i][i], 1)
            for j in range(3):
                if i != j:
                    self.assertEqual(cm[i][j], 0)

    def test_custom_prefix(self):
        result = save_confusion_matrix_artifacts(
            _Y_TRUE, _Y_PRED, _CLASS_NAMES, self._out, prefix="my_cm"
        )
        self.assertIn("my_cm", result["raw_csv_path"])
        self.assertTrue(Path(result["raw_csv_path"]).exists())

    def test_two_class_problem(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = save_confusion_matrix_artifacts(y_true, y_pred, ["Neg", "Pos"], self._out)
        cm = result["confusion_matrix"]
        self.assertEqual(len(cm), 2)
        self.assertEqual(cm[0][0], 1)  # TN
        self.assertEqual(cm[1][1], 1)  # TP


if __name__ == "__main__":
    unittest.main()
