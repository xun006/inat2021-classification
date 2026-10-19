import csv
import json
import tempfile
import unittest
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from baselines.d_group_training_free_baselines import (
    METHODS,
    closed_set_oscr_curve,
    evaluate_scores,
    percent_string,
    risk_coverage_curve,
    score_tensors,
    validate_d_checkpoint_config,
    verify_best_selection,
    write_metrics,
)


class DGroupBaselineTests(unittest.TestCase):
    def test_score_definitions_and_directions(self):
        logits = torch.tensor([[4.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        scores = score_tensors(logits)
        self.assertEqual(tuple(scores), METHODS)
        for values in scores.values():
            self.assertEqual(tuple(values.shape), (2,))
            self.assertTrue(torch.isfinite(values).all())
        for method in ("Max-Sigmoid", "Sigmoid Margin", "Raw-logit Margin",
                       "DOCTOR", "L2Norm-MaxLogit"):
            self.assertGreater(float(scores[method][1]), float(scores[method][0]))

        top1 = logits[0, 0]
        expected_ovr = (
            F.softplus(-top1) + F.softplus(logits[0, 1:]).sum()
        ) / logits.shape[1]
        self.assertAlmostEqual(float(scores["OVR Self-BCE"][0]),
                               float(expected_ovr), places=6)
        probabilities = torch.softmax(logits[0], 0)
        collision = probabilities.square().sum()
        self.assertAlmostEqual(float(scores["DOCTOR"][0]),
                               float((1 - collision) / collision), places=6)
        expected_l2 = -float(logits[0].max() / torch.linalg.vector_norm(logits[0]))
        self.assertAlmostEqual(float(scores["L2Norm-MaxLogit"][0]), expected_l2, places=6)

    def test_zero_logits_are_finite(self):
        scores = score_tensors(torch.zeros(2, 3))
        for values in scores.values():
            self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.equal(scores["L2Norm-MaxLogit"], torch.zeros(2)))

    def test_max_sigmoid_and_max_logit_have_same_ranking_without_ties(self):
        logits = torch.tensor([[3.0, 0.0], [1.0, -2.0], [2.0, 1.0]])
        max_sigmoid_error = score_tensors(logits)["Max-Sigmoid"].numpy()
        max_logit_error = -logits.max(dim=1).values.numpy()
        np.testing.assert_array_equal(np.argsort(max_sigmoid_error), np.argsort(max_logit_error))

    def test_perfect_failure_ranking(self):
        errors = np.asarray([0, 0, 1, 1])
        values = np.asarray([0.0, 0.1, 0.9, 1.0])
        scores = OrderedDict((method, values.copy()) for method in METHODS)
        rows, _, _ = evaluate_scores(errors, scores)
        for row in rows:
            self.assertAlmostEqual(row["auroc"], 1.0)
            self.assertAlmostEqual(row["error_auprc"], 1.0)
            self.assertAlmostEqual(row["fpr_at_95_tpr"], 0.0)
            self.assertAlmostEqual(row["oscr"], 0.5)

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

    def test_percentage_rounding_is_half_up_and_csv_has_two_decimals(self):
        self.assertEqual(percent_string(0.914884), "91.49")
        self.assertEqual(percent_string(0.12345), "12.35")
        self.assertEqual(Decimal(percent_string(0.0)), Decimal("0.00"))
        row = {"method": METHODS[0], "auroc": 0.914884, "error_auprc": 0.12345,
               "fpr_at_95_tpr": 0.2, "aurc": 0.01, "oscr": 0.5}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "metrics.csv"
            write_metrics(path, [row])
            with path.open(newline="", encoding="utf-8") as handle:
                written = next(csv.DictReader(handle))
            self.assertEqual(written["auroc"], "91.49")
            self.assertEqual(written["error_auprc"], "12.35")
            self.assertEqual(written["fpr_at_95_tpr"], "20.00")

    def test_d_protocol_validation_accepts_legacy_metadata(self):
        cfg = {"loss": "l2", "seed": 42, "epochs": 20, "batch_size": 32,
               "rank": 8, "alpha": 16, "lora_dropout": 0.0,
               "bce_reduction": "mean", "workers": 8}
        validate_d_checkpoint_config(cfg, {"config": cfg.copy()}, Path("best.pt"))

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
