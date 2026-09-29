import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .evaluate_failure import run, select_threshold
from .metrics import failure_detection_metrics


class FailureMetricTests(unittest.TestCase):
    def test_perfect_and_reverse_rankings(self):
        error = np.array([0, 0, 1, 1])
        perfect, _ = failure_detection_metrics(error, [.9, .8, .2, .1])
        reverse, _ = failure_detection_metrics(error, [.2, .1, .9, .8])
        self.assertEqual(perfect["auroc_error"], 1.0)
        self.assertEqual(perfect["aupr_error"], 1.0)
        self.assertEqual(perfect["fpr_at_95_tpr"], 0.0)
        self.assertAlmostEqual(perfect["aurc"], (0 + 0 + 1 / 3 + 1 / 2) / 4)
        self.assertEqual(reverse["auroc_error"], 0.0)
        self.assertEqual(reverse["fpr_at_95_tpr"], 1.0)

    def test_ties_and_degenerate_labels(self):
        a, _ = failure_detection_metrics([0, 1], [.5, .5])
        b, _ = failure_detection_metrics([1, 0], [.5, .5])
        self.assertEqual(a, b)
        self.assertEqual(a["auroc_error"], .5)
        for error in ([0, 0], [1, 1]):
            metrics, _ = failure_detection_metrics(error, [.9, .1])
            self.assertIsNone(metrics["auroc_error"])
            self.assertIsNone(metrics["aupr_error"])
            self.assertIsNone(metrics["fpr_at_95_tpr"])
            self.assertIsNotNone(metrics["aurc"])

    def test_invalid_metric_inputs(self):
        for error, confidence in (([], []), ([0], [np.nan]), ([0], [1.1])):
            with self.assertRaises(ValueError):
                failure_detection_metrics(error, confidence)

    def test_threshold_uses_whole_tie_group(self):
        error = np.array([1, 1, 0, 0], dtype=bool)
        confidence = np.array([.1, .2, .2, .9])
        tau, boundary, point = select_threshold(error, confidence, .75)
        self.assertEqual(boundary, .2)
        self.assertGreater(tau, boundary)
        self.assertEqual((point["tp"], point["fp"]), (2, 1))
        self.assertGreaterEqual(point["error_recall"], .75)


class FailureEvaluationCommandTests(unittest.TestCase):
    @staticmethod
    def make_export(root, split, rows, checkpoint="checkpoint-a"):
        export = root / (split + "_export")
        export.mkdir()
        with (export / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "correct", "max_sigmoid_probability"],
            )
            writer.writeheader()
            for index, (correct, confidence) in enumerate(rows):
                writer.writerow(
                    {
                        "sample_id": "{}/{}.jpg".format(index % 2, index),
                        "correct": correct,
                        "max_sigmoid_probability": confidence,
                    }
                )
        (export / "complete.json").write_text(
            json.dumps({"rows": len(rows)}), encoding="utf-8"
        )
        (export / "provenance.json").write_text(
            json.dumps(
                {
                    "checkpoint_sha256": checkpoint,
                    "pretrained_sha256": "pretrained-a",
                    "split_dir": "/data/{}".format(split),
                    "class_to_idx": {"a": 0, "b": 1},
                    "config": {"loss": "l2", "seed": 42},
                }
            ),
            encoding="utf-8",
        )
        return export

    def test_end_to_end_and_test_data_cannot_change_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = self.make_export(
                root,
                "detector_calibration",
                [(0, .1), (0, .2), (1, .2), (1, .9)],
            )
            official = self.make_export(
                root,
                "official_val",
                [(1, .9), (0, .8), (1, .7), (0, .1)],
            )
            run(calibration, official, root / "result-a", .95)
            first = json.loads(
                (root / "result-a" / "threshold.json").read_text(encoding="utf-8")
            )

            with (official / "samples.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["correct"] = "0"
            rows[0]["max_sigmoid_probability"] = "0.01"
            with (official / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            run(calibration, official, root / "result-b", .95)
            second = json.loads(
                (root / "result-b" / "threshold.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first, second)
            for name in ("metrics.json", "operating_point.json", "summary.csv", "summary.md"):
                self.assertTrue((root / "result-a" / name).is_file())

    def test_model_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = self.make_export(
                root, "detector_calibration", [(0, .1), (1, .9)]
            )
            official = self.make_export(
                root, "official_val", [(0, .2), (1, .8)], checkpoint="checkpoint-b"
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                run(calibration, official, root / "result", .95)


if __name__ == "__main__":
    unittest.main()
