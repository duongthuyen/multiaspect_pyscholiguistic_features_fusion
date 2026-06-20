"""
Tests for pure / nearly-pure functions in scripts/models/fusion/train.py.

No real data files are required. Tiny fake models and DataLoaders are used
to verify the training loop helpers without any heavy computation.
"""

import unittest

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts.training.fusion_train import (
    _predict,
    _scale_hc,
    accuracy,
    run_epoch,
    set_seed,
)


class SetSeedTests(unittest.TestCase):
    def test_torch_determinism(self):
        set_seed(42)
        a = torch.randn(4, 4)
        set_seed(42)
        b = torch.randn(4, 4)
        self.assertTrue(torch.equal(a, b))

    def test_numpy_determinism(self):
        set_seed(0)
        a = np.random.rand(3, 3)
        set_seed(0)
        b = np.random.rand(3, 3)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_results(self):
        set_seed(1)
        a = torch.randn(10)
        set_seed(2)
        b = torch.randn(10)
        self.assertFalse(torch.equal(a, b))


class AccuracyTests(unittest.TestCase):
    def test_all_correct(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        labels = torch.tensor([0, 1])
        self.assertAlmostEqual(accuracy(logits, labels), 1.0)

    def test_all_wrong(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        labels = torch.tensor([1, 2])
        self.assertAlmostEqual(accuracy(logits, labels), 0.0)

    def test_half_correct(self):
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
        labels = torch.tensor([0, 0, 0, 0])
        # Preds: [0, 1, 0, 1]; only 0 and 2 are correct → 0.5
        self.assertAlmostEqual(accuracy(logits, labels), 0.5)

    def test_returns_float(self):
        logits = torch.randn(5, 3)
        labels = torch.randint(0, 3, (5,))
        result = accuracy(logits, labels)
        self.assertIsInstance(result, float)


class ScaleHcTests(unittest.TestCase):
    def test_fit_transform_changes_statistics(self):
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        scaled = _scale_hc(scaler, data, fit=True)
        # After StandardScaler the mean of each column is approximately 0
        self.assertAlmostEqual(float(scaled[:, 0].mean()), 0.0, places=5)
        self.assertAlmostEqual(float(scaled[:, 1].mean()), 0.0, places=5)

    def test_transform_uses_fitted_statistics(self):
        train = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        test = torch.tensor([[1.0, 1.0]])
        scaler = StandardScaler()
        _scale_hc(scaler, train, fit=True)
        scaled_test = _scale_hc(scaler, test, fit=False)
        # mean of train columns is 1.0; test value 1.0 → scaled ~= 0.0
        self.assertAlmostEqual(float(scaled_test[0, 0]), 0.0, places=4)

    def test_returns_float32_tensor(self):
        data = torch.randn(4, 3)
        scaler = StandardScaler()
        scaled = _scale_hc(scaler, data, fit=True)
        self.assertIsInstance(scaled, torch.Tensor)
        self.assertEqual(scaled.dtype, torch.float32)

    def test_output_shape_preserved(self):
        data = torch.randn(8, 5)
        scaler = StandardScaler()
        scaled = _scale_hc(scaler, data, fit=True)
        self.assertEqual(scaled.shape, data.shape)


class _TinyFusionModel(nn.Module):
    """Minimal model with the same (semantic, affective, handcrafted) signature."""

    def __init__(self, in_dim: int, num_labels: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_labels)

    def forward(
        self,
        semantic: torch.Tensor,
        affective: torch.Tensor,
        handcrafted: torch.Tensor,
    ) -> torch.Tensor:
        return self.fc(torch.cat([semantic, affective, handcrafted], dim=1))

    def training_step(self, semantic, affective, handcrafted, labels, criterion):
        logits = self.forward(semantic, affective, handcrafted)
        return criterion(logits, labels), logits, torch.empty(logits.shape[0], 3)


def _make_loader(n_rows: int = 8, sem_dim: int = 4, aff_dim: int = 2, hc_dim: int = 2, num_labels: int = 3):
    sem = torch.randn(n_rows, sem_dim)
    aff = torch.randn(n_rows, aff_dim)
    hc = torch.randn(n_rows, hc_dim)
    labels = torch.randint(0, num_labels, (n_rows,))
    ds = TensorDataset(sem, aff, hc, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)


class RunEpochTests(unittest.TestCase):
    _IN_DIM = 8  # sem(4) + aff(2) + hc(2)
    _NUM_LABELS = 3

    def _model_and_loader(self):
        model = _TinyFusionModel(self._IN_DIM, self._NUM_LABELS)
        loader = _make_loader(n_rows=8, sem_dim=4, aff_dim=2, hc_dim=2, num_labels=self._NUM_LABELS)
        return model, loader

    def test_training_loss_is_finite(self):
        model, loader = self._model_and_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")
        loss, acc = run_epoch(model, loader, criterion, optimizer, device)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(acc))

    def test_eval_mode_when_no_optimizer(self):
        model, loader = self._model_and_loader()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        # Run without optimizer — should not update weights
        params_before = {k: v.clone() for k, v in model.state_dict().items()}
        run_epoch(model, loader, criterion, None, device)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, params_before[k]), msg=f"Param {k} changed during eval")

    def test_training_updates_weights(self):
        model, loader = self._model_and_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        device = torch.device("cpu")
        params_before = {k: v.clone() for k, v in model.state_dict().items()}
        run_epoch(model, loader, criterion, optimizer, device)
        any_changed = any(
            not torch.equal(v, params_before[k]) for k, v in model.state_dict().items()
        )
        self.assertTrue(any_changed)

    def test_returns_mean_loss_and_accuracy(self):
        model, loader = self._model_and_loader()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        loss, acc = run_epoch(model, loader, criterion, None, device)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


class PredictTests(unittest.TestCase):
    _IN_DIM = 8
    _NUM_LABELS = 3

    def test_prediction_shapes(self):
        model = _TinyFusionModel(self._IN_DIM, self._NUM_LABELS)
        model.eval()
        loader = _make_loader(n_rows=10, sem_dim=4, aff_dim=2, hc_dim=2, num_labels=self._NUM_LABELS)
        device = torch.device("cpu")
        preds, labels = _predict(model, loader, device)
        self.assertEqual(preds.shape, (10,))
        self.assertEqual(labels.shape, (10,))

    def test_predictions_within_label_range(self):
        model = _TinyFusionModel(self._IN_DIM, self._NUM_LABELS)
        loader = _make_loader(n_rows=8, sem_dim=4, aff_dim=2, hc_dim=2, num_labels=self._NUM_LABELS)
        device = torch.device("cpu")
        preds, _ = _predict(model, loader, device)
        self.assertTrue(np.all(preds >= 0))
        self.assertTrue(np.all(preds < self._NUM_LABELS))

    def test_does_not_require_grad(self):
        # _predict should work even on a model that has no gradient context
        model = _TinyFusionModel(self._IN_DIM, self._NUM_LABELS)
        for p in model.parameters():
            p.requires_grad_(False)
        loader = _make_loader(n_rows=4, sem_dim=4, aff_dim=2, hc_dim=2, num_labels=self._NUM_LABELS)
        preds, labels = _predict(model, loader, torch.device("cpu"))
        self.assertEqual(len(preds), 4)


if __name__ == "__main__":
    unittest.main()
