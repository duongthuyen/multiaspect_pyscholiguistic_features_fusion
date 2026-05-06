"""
Tests for output-writing functions in scripts/models/evaluate_fusion.py.

Model loading and inference are NOT tested here (they require a trained
checkpoint on disk). Only the pure serialisation and formatting logic
is covered.
"""

import json
import tempfile
import unittest
from pathlib import Path

import scripts.config as cfg
from scripts.models.evaluate_fusion import save_evaluation

_ORIG_RESULTS_DIR = cfg.RESULTS_DIR


def _fake_result(input_config: str = "fused", model_type: str = "concat", split: str = "test") -> dict:
    """Minimal result dict matching the structure produced by evaluate()."""
    class_names = [cfg.ID_TO_CLASS[i] for i in range(cfg.NUM_LABELS)]
    return {
        "model_type": model_type,
        "input_config": input_config,
        "split": split,
        "accuracy": 0.812345,
        "macro_f1": 0.791234,
        "weighted_f1": 0.808765,
        "per_class": {
            cls: {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 50}
            for cls in class_names
        },
        "class_names": class_names,
        "confusion_matrix": {
            "confusion_matrix": [[10] * len(class_names)] * len(class_names),
            "raw_csv_path": "cm.csv",
            "raw_json_path": "cm.json",
            "plot_path": "cm.png",
        },
    }


class SaveEvaluationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg.RESULTS_DIR = Path(self._tmp.name)

    def tearDown(self):
        cfg.RESULTS_DIR = _ORIG_RESULTS_DIR
        self._tmp.cleanup()

    def _run(self, **kwargs) -> dict:
        result = _fake_result(**kwargs)
        save_evaluation(result)
        return result

    def test_creates_metrics_json(self):
        result = self._run()
        from scripts.utils.outputs import evaluation_dir
        json_path = evaluation_dir("fused", "concat") / "test" / "metrics.json"
        self.assertTrue(json_path.exists(), msg=f"Expected {json_path}")

    def test_metrics_json_content(self):
        result = self._run()
        from scripts.utils.outputs import evaluation_dir
        json_path = evaluation_dir("fused", "concat") / "test" / "metrics.json"
        with open(json_path) as f:
            data = json.load(f)
        self.assertAlmostEqual(data["accuracy"], result["accuracy"])
        self.assertEqual(data["split"], "test")
        self.assertEqual(data["model_type"], "concat")

    def test_creates_summary_txt(self):
        result = self._run()
        from scripts.utils.outputs import evaluation_dir
        txt_path = evaluation_dir("fused", "concat") / "test" / "summary.txt"
        self.assertTrue(txt_path.exists())

    def test_summary_txt_contains_metrics(self):
        result = self._run()
        from scripts.utils.outputs import evaluation_dir
        txt_path = evaluation_dir("fused", "concat") / "test" / "summary.txt"
        text = txt_path.read_text(encoding="utf-8")
        self.assertIn("0.8123", text)
        self.assertIn("concat", text)
        self.assertIn("fused", text)

    def test_different_split_creates_separate_directory(self):
        self._run(split="val")
        from scripts.utils.outputs import evaluation_dir
        val_path = evaluation_dir("fused", "concat") / "val" / "metrics.json"
        self.assertTrue(val_path.exists())

    def test_different_model_creates_separate_directory(self):
        self._run(model_type="gated")
        from scripts.utils.outputs import evaluation_dir
        json_path = evaluation_dir("fused", "gated") / "test" / "metrics.json"
        self.assertTrue(json_path.exists())

    def test_all_class_names_in_summary(self):
        result = self._run()
        from scripts.utils.outputs import evaluation_dir
        txt = (evaluation_dir("fused", "concat") / "test" / "summary.txt").read_text()
        for cls_name in cfg.ID_TO_CLASS.values():
            self.assertIn(cls_name, txt)


if __name__ == "__main__":
    unittest.main()
