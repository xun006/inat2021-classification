"""Small deterministic tests for formulas that do not require PlantCLEF files."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from math_core import (
    build_distance_matrix,
    expected_tie_aware_aurc,
    failure_metrics,
    relative_uncertainty,
)
from fit_select import estimate_moments, score_cache


class RelUCoreTests(unittest.TestCase):
    def test_distance_matches_author_formula(self) -> None:
        correct = torch.tensor([[0.5, 0.1, 0.0], [0.1, 0.4, 0.1], [0.0, 0.1, 0.5]])
        error = torch.tensor([[0.3, 0.4, 0.1], [0.4, 0.2, 0.3], [0.1, 0.3, 0.2]])
        lbd = 0.6
        expected = torch.relu(torch.tril(lbd * error - (1 - lbd) * correct, diagonal=-1))
        expected = expected + expected.T
        expected = expected / expected.norm()
        actual, diagnostic = build_distance_matrix(correct, error, lbd)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(diagnostic.fallback_to_gini)
        self.assertEqual(float(torch.diagonal(actual).abs().sum()), 0.0)

    def test_score_avoids_batch_square_but_matches_definition(self) -> None:
        probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]])
        distance = torch.tensor([[0.0, 0.5, 0.2], [0.5, 0.0, 0.3], [0.2, 0.3, 0.0]])
        expected = torch.diag(probabilities @ distance @ probabilities.T)
        torch.testing.assert_close(relative_uncertainty(probabilities, distance), expected)

    def test_gini_fallback(self) -> None:
        moment = torch.eye(3)
        distance, diagnostic = build_distance_matrix(moment, moment, 0.5)
        self.assertTrue(diagnostic.fallback_to_gini)
        self.assertTrue(torch.all(distance[~torch.eye(3, dtype=torch.bool)] > 0))

    def test_tie_aware_aurc_is_order_invariant(self) -> None:
        error_a = np.array([0, 1, 0, 1])
        error_b = np.array([1, 0, 1, 0])
        tied = np.ones(4)
        self.assertAlmostEqual(
            expected_tie_aware_aurc(error_a, tied),
            expected_tie_aware_aurc(error_b, tied),
        )

    def test_metrics_use_error_as_positive(self) -> None:
        error = np.array([0, 0, 1, 1])
        perfect_score = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = failure_metrics(error, perfect_score)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["error_auprc"], 1.0)
        self.assertEqual(metrics["fpr_at_95_error_tpr"], 0.0)

    def test_streamed_moments_and_scores_match_direct_computation(self) -> None:
        logits = np.array([
            [2.0, 0.0, -1.0],
            [0.0, 1.0, 0.5],
            [-0.5, 0.0, 1.5],
            [0.2, 0.1, 0.0],
            [0.0, 1.2, -0.4],
        ], dtype=np.float32)
        errors = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        temperature = 0.75
        correct, wrong, n_correct, n_wrong = estimate_moments(
            logits, errors, temperature, classes=3, batch_size=2, device="cpu"
        )
        probabilities = torch.softmax(torch.from_numpy(logits) / temperature, dim=1)
        direct_correct = probabilities[errors == 0].T @ probabilities[errors == 0] / 3
        direct_wrong = probabilities[errors == 1].T @ probabilities[errors == 1] / 2
        torch.testing.assert_close(correct, direct_correct)
        torch.testing.assert_close(wrong, direct_wrong)
        self.assertEqual((n_correct, n_wrong), (3, 2))
        distance, _ = build_distance_matrix(correct, wrong, 0.7)
        streamed = score_cache(logits, temperature, distance, batch_size=2, device="cpu")
        direct_scores = relative_uncertainty(probabilities, distance).numpy()
        np.testing.assert_allclose(streamed, direct_scores, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
