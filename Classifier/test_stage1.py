"""CPU tests for the unified formal classifier engine."""
import tempfile
import unittest
from pathlib import Path

import torch

from .checkpoint import load, restore_model, save
from .engine import learning_rate_factor, loss_values, optimizer_step, resolve_config
from .model import build_model
from .protocol import GROUPS


class UnifiedProtocolTests(unittest.TestCase):
    def test_schedules_and_group_definitions(self):
        for group in "ABCD":
            cfg = resolve_config(Path("Classifier/config.json"), group, 42)
            self.assertEqual((cfg["epochs"], cfg["lr_warmup"]), (20, 2))
        for group in "GHI":
            cfg = resolve_config(Path("Classifier/config.json"), group, 42)
            self.assertEqual((cfg["epochs"], cfg["lr_warmup"]), (30, 3))
            self.assertEqual(cfg["tuning_mode"], "full")
        self.assertEqual(tuple(GROUPS), ("A", "B", "C", "D", "G", "H", "I"))

    def test_abcd_historical_training_contract(self):
        expected = {
            "A": ("head_only", "ce", "max_softmax"),
            "B": ("head_only", "l2", "max_sigmoid"),
            "C": ("lora_head", "ce", "max_softmax"),
            "D": ("lora_head", "l2", "max_sigmoid"),
        }
        for group, values in expected.items():
            cfg = resolve_config(Path("Classifier/config.json"), group, 42)
            self.assertEqual(
                (cfg["tuning_mode"], cfg["loss"], cfg["native_confidence"]), values)
            self.assertEqual(cfg["initialization"], "plantclef")
            self.assertEqual(cfg["lr"], 1e-4)
            self.assertEqual(cfg["head_lr"], 1e-3)

    def test_learning_rate_horizons(self):
        abcd = resolve_config(Path("Classifier/config.json"), "A", 42)
        ghi = resolve_config(Path("Classifier/config.json"), "G", 42)
        self.assertEqual(learning_rate_factor(abcd, 0), 0)
        self.assertEqual(learning_rate_factor(abcd, 2), 1)
        self.assertAlmostEqual(learning_rate_factor(abcd, 20), 0)
        self.assertEqual(learning_rate_factor(ghi, 3), 1)
        self.assertAlmostEqual(learning_rate_factor(ghi, 30), 0)


class UnifiedEngineTests(unittest.TestCase):
    def test_loss_selection(self):
        logits = torch.tensor([[2.0, -1.0], [-0.5, 1.0]])
        labels = torch.tensor([0, 1])
        ce = loss_values(logits, labels, {"loss": "ce", "bce_reduction": "mean"})
        bce = loss_values(logits, labels, {"loss": "l2", "bce_reduction": "mean"})
        self.assertEqual(float(ce[0]), float(ce[1]))
        self.assertEqual(float(bce[0]), float(bce[2]))

    def test_scaler_skips_overflow(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
        scaler.scale(model(torch.ones(2, 2)).sum()).backward()
        model.weight.grad[0, 0] = float("inf")
        event = optimizer_step(model, optimizer, scaler, 1.0)
        self.assertTrue(event["skipped"])
        self.assertEqual(event["scale_after"], 4.0)

    def test_three_trainable_scopes(self):
        base = dict(model="vit_tiny_patch16", image_size=32, num_classes=2,
                    global_pool=True, rank=2, alpha=4, lora_dropout=0.0,
                    initialization="random")
        head, _ = build_model({**base, "tuning_mode": "head_only"}, initialize=False)
        lora, _ = build_model({**base, "tuning_mode": "lora_head"}, initialize=False)
        full, _ = build_model({**base, "tuning_mode": "full"}, initialize=False)
        self.assertTrue(all(name.startswith("head.") for name, parameter in head.named_parameters()
                            if parameter.requires_grad))
        self.assertTrue(any(".a.weight" in name for name, parameter in lora.named_parameters()
                            if parameter.requires_grad))
        self.assertTrue(all(parameter.requires_grad for parameter in full.parameters()))

    def test_checkpoint_round_trip(self):
        cfg = dict(model="vit_tiny_patch16", image_size=32, num_classes=2,
                   global_pool=True, rank=2, alpha=4, lora_dropout=0.0,
                   initialization="random", tuning_mode="head_only")
        model, _ = build_model(cfg, initialize=False)
        optimizer = torch.optim.AdamW(model.head.parameters())
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "last.pt"
            save(path, "last", model, cfg, 0, 0.5, {"a": 0, "b": 1},
                 {"train_counts": [1, 1]}, "random_full_state:seed=42", optimizer, scaler)
            saved = load(path)
            restored, _ = build_model(cfg, initialize=False)
            restore_model(restored, saved)
            torch.testing.assert_close(model.head.weight, restored.head.weight)


if __name__ == "__main__":
    unittest.main()
