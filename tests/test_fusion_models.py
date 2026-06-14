import unittest

import torch

from scripts import config
from scripts.models.fusion import (
    ClassAwareGatedFusion,
    ContentGatedFusion,
    LoadBalancedFusion,
    PerClassGatedFusion,
    build_gated_model,
)
from scripts.models.fusion.feature_loader import GROUP_SUBFEATURES


class FusionModelTests(unittest.TestCase):
    def _inputs(self, batch_size=5):
        return (
            torch.randn(batch_size, config.SEMANTIC_DIM),
            torch.randn(batch_size, config.AFFECTIVE_DIM),
            torch.randn(batch_size, config.HANDCRAFTED_DIM),
        )

    def _assert_forward_backward(self, model):
        semantic, affective, handcrafted = self._inputs()
        logits = model(semantic, affective, handcrafted)
        self.assertEqual(logits.shape, (semantic.shape[0], config.NUM_LABELS))
        self.assertTrue(torch.isfinite(logits).all())
        loss = logits.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(grad is not None for grad in grads))
        self.assertTrue(
            all(torch.isfinite(grad).all() for grad in grads if grad is not None)
        )

    def test_content_gated_forward_backward(self):
        self._assert_forward_backward(ContentGatedFusion())

    def test_gated_branch_representations_equal_projection_dim(self):
        model = ContentGatedFusion()
        semantic, affective, handcrafted = self._inputs(batch_size=3)
        sem_repr, aff_repr, hand_repr = model.get_branch_representations(
            semantic, affective, handcrafted
        )
        for name, repr_ in [("semantic", sem_repr), ("affective", aff_repr), ("handcrafted", hand_repr)]:
            self.assertEqual(
                repr_.shape,
                (3, config.GATED_PROJECTION_DIM),
                msg=f"{name} branch shape mismatch",
            )

    def test_gated_return_gates_shape(self):
        model = ContentGatedFusion()
        semantic, affective, handcrafted = self._inputs(batch_size=4)
        logits, gates = model(semantic, affective, handcrafted, return_gates=True)
        self.assertEqual(logits.shape, (4, config.NUM_LABELS))
        self.assertEqual(gates.shape, (4, 3))

    def test_all_gated_variants_forward_backward(self):
        for cls in (
            ContentGatedFusion,
            ClassAwareGatedFusion,
            LoadBalancedFusion,
            PerClassGatedFusion,
        ):
            with self.subTest(cls=cls.__name__):
                self._assert_forward_backward(cls())

    def test_build_gated_model_variants(self):
        expected = {
            "content_gate": ContentGatedFusion,
            "class_aware": ClassAwareGatedFusion,
            "load_balance": LoadBalancedFusion,
            "per_class_gate": PerClassGatedFusion,
        }
        for variant, cls in expected.items():
            with self.subTest(variant=variant):
                self.assertIsInstance(build_gated_model(variant), cls)

    def test_build_gated_model_invalid_variant_raises(self):
        with self.assertRaises(ValueError):
            build_gated_model("invalid")

    def test_feature_loader_group_registry(self):
        self.assertEqual(GROUP_SUBFEATURES["semantic"], ["mental_roberta"])
        self.assertEqual(GROUP_SUBFEATURES["affective"], ["goemotions", "vad", "vader"])

    def test_split_aware_feature_loader_path_error(self):
        from scripts.models.fusion.feature_loader import load_subextractor_features

        with self.assertRaises(FileNotFoundError) as ctx:
            load_subextractor_features("semantic", "mental_roberta", split="missing")
        msg = str(ctx.exception)
        self.assertIn("mental_roberta.parquet", msg)
        self.assertIn("missing", msg)


if __name__ == "__main__":
    unittest.main()
