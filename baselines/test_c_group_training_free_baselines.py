import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from baselines.c_group_training_free_baselines import (
    METHODS,
    closed_set_oscr_curve,
    evaluate_scores,
    misclassification_oscr_md_curve,
    risk_coverage_curve,
    score_tensors,
    verify_best_selection,
)


class CGroupBaselineTests(unittest.TestCase):
    def test_score_definitions_and_directions(self):
        logits = torch.tensor([[4.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        scores = score_tensors(logits)
        self.assertEqual(tuple(scores), METHODS)
        for values in scores.values():
            self.assertEqual(tuple(values.shape), (2,))
            self.assertTrue(torch.isfinite(values).all())
        # The uniform prediction is more uncertain/error-like for these five scores.
        for method in ("MSP", "Probability Margin", "Entropy", "DOCTOR",
                       "Raw-logit Margin"):
            self.assertGreater(float(scores[method][1]), float(scores[method][0]))
        expected_g = torch.softmax(logits[0], 0).square().sum()
        self.assertAlmostEqual(
            float(scores["DOCTOR"][0]), float((1 - expected_g) / expected_g), places=6
        )
        self.assertAlmostEqual(float(scores["Energy"][0]),
                               float(-torch.logsumexp(logits[0], 0)), places=6)

    def test_perfect_failure_ranking(self):
        errors = np.asarray([0, 0, 1, 1])
        values = np.asarray([0.0, 0.1, 0.9, 1.0])
        scores = OrderedDict((method, values.copy()) for method in METHODS)
        rows, _, _, _ = evaluate_scores(errors, scores)
        for row in rows:
            self.assertAlmostEqual(row["auroc"], 1.0)
            self.assertAlmostEqual(row["error_auprc"], 1.0)
            self.assertAlmostEqual(row["fpr_at_95_tpr"], 0.0)
            self.assertAlmostEqual(row["oscr"], 0.5)
            self.assertAlmostEqual(row["oscr_md"], 0.5)

    def test_tie_aware_aurc_is_permutation_invariant(self):
        errors_a = np.asarray([0, 1, 0, 1])
        errors_b = np.asarray([1, 0, 0, 1])
        scores = np.asarray([0.0, 0.0, 1.0, 1.0])
        _, risk_a, aurc_a = risk_coverage_curve(errors_a, scores)
        _, risk_b, aurc_b = risk_coverage_curve(errors_b, scores)
        np.testing.assert_allclose(risk_a, risk_b)
        self.assertAlmostEqual(aurc_a, aurc_b)

    def test_closed_set_oscr_identity(self):
        errors = np.asarray([0, 1, 0, 1, 0])
        scores = np.asarray([0.2, 0.8, 0.2, 0.7, 0.1])
        _, _, oscr = closed_set_oscr_curve(errors, scores)
        accuracy = float((errors == 0).mean())
        self.assertAlmostEqual(oscr, accuracy * roc_auc_score(errors, scores))

    def test_oscr_md_uses_all_unique_confidence_thresholds(self):
        errors = np.asarray([0, 1, 0, 1, 0])
        error_scores = np.asarray([0.1, 0.8, 0.1, 0.7, 0.2])
        thresholds, ear, mccr, oscr_md = misclassification_oscr_md_curve(
            errors, error_scores
        )

        # Four unique confidence values plus the reject-all point at +infinity.
        self.assertEqual(len(thresholds), 5)
        self.assertTrue(np.isposinf(thresholds[0]))
        np.testing.assert_allclose(thresholds[1:], [-0.1, -0.2, -0.7, -0.8])
        np.testing.assert_allclose(ear, [0.0, 0.0, 0.0, 0.5, 1.0])
        np.testing.assert_allclose(mccr, [0.0, 0.4, 0.6, 0.6, 0.6])
        self.assertAlmostEqual(oscr_md, 0.6)

    def test_oscr_md_ties_are_permutation_invariant(self):
        scores = np.asarray([0.2, 0.2, 0.8, 0.8])
        _, ear_a, mccr_a, area_a = misclassification_oscr_md_curve(
            np.asarray([0, 1, 0, 1]), scores
        )
        _, ear_b, mccr_b, area_b = misclassification_oscr_md_curve(
            np.asarray([1, 0, 1, 0]), scores
        )
        np.testing.assert_allclose(ear_a, ear_b)
        np.testing.assert_allclose(mccr_a, mccr_b)
        self.assertAlmostEqual(area_a, area_b)

    def test_best_checkpoint_history_audit_uses_earliest_tie(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            rows = [
                {"epoch": 0, "val": {"top1": 0.70}},
                {"epoch": 1, "val": {"top1": 0.80}},
                {"epoch": 2, "val": {"top1": 0.80}},
            ]
            (root / "history.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            audit = verify_best_selection(checkpoint, {"epoch": 1, "best": 0.80})
            self.assertTrue(audit["verified"])
            self.assertEqual(audit["selected_epoch_one_based"], 2)


if __name__ == "__main__":
    unittest.main()
