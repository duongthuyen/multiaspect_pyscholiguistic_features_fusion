"""Tests for scripts/models/fusion/blocks.py — ProjectionBlock and ClassifierHead."""

import unittest

import torch

from scripts import config
from scripts.models.fusion.blocks import ClassifierHead, ProjectionBlock


class ProjectionBlockTests(unittest.TestCase):
    def _forward(self, activation: str, batch_size: int = 4) -> torch.Tensor:
        block = ProjectionBlock(input_dim=16, output_dim=8, activation=activation, dropout=0.0)
        block.eval()
        return block(torch.randn(batch_size, 16))

    def test_gelu_output_shape(self):
        out = self._forward("gelu")
        self.assertEqual(out.shape, (4, 8))

    def test_tanh_output_shape(self):
        out = self._forward("tanh")
        self.assertEqual(out.shape, (4, 8))

    def test_output_is_finite(self):
        self.assertTrue(torch.isfinite(self._forward("gelu")).all())
        self.assertTrue(torch.isfinite(self._forward("tanh")).all())

    def test_invalid_activation_raises(self):
        block = ProjectionBlock(input_dim=4, output_dim=4, activation="relu")
        with self.assertRaises(ValueError):
            block(torch.randn(2, 4))

    def test_gradients_flow(self):
        block = ProjectionBlock(input_dim=8, output_dim=4, activation="gelu", dropout=0.0)
        out = block(torch.randn(3, 8))
        out.sum().backward()
        self.assertTrue(
            any(p.grad is not None for p in block.parameters() if p.requires_grad)
        )

    def test_dropout_zeroes_in_training(self):
        # With dropout=1.0 and training mode all outputs should be zero
        block = ProjectionBlock(input_dim=8, output_dim=4, activation="gelu", dropout=1.0)
        block.train()
        out = block(torch.randn(10, 8))
        self.assertTrue((out == 0).all())

    def test_batch_size_one(self):
        block = ProjectionBlock(input_dim=6, output_dim=3, activation="tanh", dropout=0.0)
        out = block(torch.randn(1, 6))
        self.assertEqual(out.shape, (1, 3))


class ClassifierHeadTests(unittest.TestCase):
    def test_output_shape_matches_num_labels(self):
        head = ClassifierHead(input_dim=64, hidden_dim=32, num_labels=6)
        head.eval()
        out = head(torch.randn(5, 64))
        self.assertEqual(out.shape, (5, 6))

    def test_output_is_finite(self):
        head = ClassifierHead(input_dim=32, hidden_dim=16, num_labels=3, dropout=0.0)
        head.eval()
        out = head(torch.randn(4, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_uses_config_num_labels_by_default(self):
        head = ClassifierHead(input_dim=16, hidden_dim=8)
        head.eval()
        out = head(torch.randn(2, 16))
        self.assertEqual(out.shape[1], config.NUM_LABELS)

    def test_gradients_flow(self):
        head = ClassifierHead(input_dim=16, hidden_dim=8, num_labels=3, dropout=0.0)
        out = head(torch.randn(3, 16))
        out.sum().backward()
        self.assertTrue(
            any(p.grad is not None for p in head.parameters() if p.requires_grad)
        )

    def test_logits_not_probabilities(self):
        # Raw logits should not sum to 1 unless coincidentally
        head = ClassifierHead(input_dim=32, hidden_dim=16, num_labels=6, dropout=0.0)
        head.eval()
        out = head(torch.randn(5, 32))
        row_sums = out.sum(dim=1)
        # Very unlikely that raw logits sum to exactly 1.0 for all rows
        self.assertFalse(torch.allclose(row_sums, torch.ones(5), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
