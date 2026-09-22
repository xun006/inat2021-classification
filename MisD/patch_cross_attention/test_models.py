#!/usr/bin/env python3
"""CPU unit tests for B1/B2/B3/M1/M2; no ViT or dataset required."""

import unittest

import torch

from models import (
    MODEL_NAMES, build_detector, effective_classifier_parameters,
    fixed_derangement, trainable_parameter_count,
)


class DetectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.weights = torch.randn(17, 1024)
        self.patches = torch.randn(3, 196, 1024)
        self.global_feature = torch.randn(3, 1024)
        self.candidates = torch.tensor([1, 5, 9])

    def test_all_output_shapes_and_finite(self):
        for name in MODEL_NAMES:
            model = build_detector(name, self.weights).eval()
            with torch.no_grad():
                output = model(
                    self.patches, self.candidates, global_feature=self.global_feature
                )
            self.assertEqual(tuple(output["error_logit"].shape), (3,))
            self.assertTrue(torch.isfinite(output["error_logit"]).all())
            if name in {"b3", "m1", "m2"}:
                attention = output["attention_weights"]
                self.assertEqual(tuple(attention.shape), (3, 4, 196))
                self.assertTrue(torch.allclose(attention.sum(-1), torch.ones(3, 4), atol=1e-5))

    def test_derangement_is_fixed_and_has_no_identity(self):
        first = fixed_derangement(4271, 42001)
        second = fixed_derangement(4271, 42001)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(first.eq(torch.arange(4271)).any())
        self.assertEqual(len(first.unique()), 4271)

    def test_detector_gradients_without_classifier_weight_gradient(self):
        model = build_detector("m2", self.weights)
        loss = model(self.patches, self.candidates)["error_logit"].square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertFalse(model.classifier_weights.requires_grad)

    def test_query_changes_attention(self):
        model = build_detector("m2", self.weights).eval()
        candidates_b = torch.tensor([2, 6, 10])
        with torch.no_grad():
            first = model(self.patches, self.candidates)
            second = model(self.patches, candidates_b)
        self.assertFalse(torch.allclose(first["attention_weights"], second["attention_weights"]))
        self.assertFalse(torch.allclose(first["attended_feature"], second["attended_feature"]))

    def test_patch_permutation_equivariance(self):
        model = build_detector("m2", self.weights).eval()
        permutation = torch.randperm(196)
        with torch.no_grad():
            original = model(self.patches, self.candidates)
            permuted = model(self.patches[:, permutation], self.candidates)
        self.assertTrue(torch.allclose(
            original["attended_feature"], permuted["attended_feature"], atol=2e-6
        ))
        self.assertTrue(torch.allclose(
            original["attention_weights"][:, :, permutation],
            permuted["attention_weights"], atol=2e-6
        ))

    def test_zero_values_prevent_query_bypass(self):
        model = build_detector("m2", self.weights).eval()
        patches = torch.zeros_like(self.patches)
        with torch.no_grad():
            first = model(patches, self.candidates)["error_logit"]
            second = model(patches, torch.tensor([2, 6, 10]))["error_logit"]
        self.assertTrue(torch.allclose(first, second, atol=1e-7))

    def test_parameter_counts_are_recordable(self):
        counts = {name: trainable_parameter_count(build_detector(name, self.weights))
                  for name in MODEL_NAMES}
        self.assertTrue(all(value > 0 for value in counts.values()))
        print("trainable parameter counts:", counts)

    def test_effective_classifier_exactly_reconstructs_eval_bn_linear(self):
        bn = torch.nn.BatchNorm1d(1024, affine=False, eps=1e-6).eval()
        bn.running_mean.copy_(torch.randn(1024))
        bn.running_var.copy_(torch.rand(1024).add_(0.01))
        linear = torch.nn.Linear(1024, 17).eval()
        effective_weight, effective_bias = effective_classifier_parameters(bn, linear)
        features = torch.randn(5, 1024)
        expected = linear(bn(features))
        actual = features.matmul(effective_weight.t()) + effective_bias
        self.assertTrue(torch.allclose(expected, actual, atol=2e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
