import json
import unittest
from pathlib import Path

from Classifier.protocol import FORMAL_SEEDS, GROUPS, apply_group, run_dir, validate_group_seed


class FormalProtocolTests(unittest.TestCase):
    def test_matrix_and_retired_groups(self):
        self.assertEqual(tuple(GROUPS), ("A", "B", "C", "D", "G", "H", "I"))
        self.assertNotIn("E", GROUPS)
        self.assertNotIn("F", GROUPS)
        self.assertEqual(GROUPS["I"]["initialization"], "imagenet")
        self.assertEqual(GROUPS["I"]["tuning_mode"], "full")

    def test_formal_seeds_and_output(self):
        self.assertEqual(FORMAL_SEEDS, (42, 43, 44))
        self.assertEqual(run_dir(Path("outputs"), "G", 43), Path("outputs/G/seed_43"))
        with self.assertRaises(ValueError):
            validate_group_seed("E", 42)
        with self.assertRaises(ValueError):
            validate_group_seed("A", 45)

    def test_family_schedules(self):
        base = json.loads(Path("Classifier/config.json").read_text(encoding="utf-8"))
        self.assertTrue(all((apply_group(base, group, 42)["epochs"],
                             apply_group(base, group, 42)["lr_warmup"]) == (20, 2)
                            for group in "ABCD"))
        self.assertTrue(all((apply_group(base, group, 42)["epochs"],
                             apply_group(base, group, 42)["lr_warmup"]) == (30, 3)
                            for group in "GHI"))


if __name__ == "__main__":
    unittest.main()
