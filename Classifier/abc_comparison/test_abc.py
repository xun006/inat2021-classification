"""CPU tests for the controlled A/B/C/E experiment."""
from __future__ import annotations

import json
import csv
import tempfile
import unittest
from unittest import mock
from argparse import Namespace
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from Classifier.abc_comparison.model import build_model
from Classifier.abc_comparison.run import (
    GROUPS, confidence_values, evaluate, loss_values, make_optimizer, resolve_config,
    train_epoch,
)
from Classifier.abc_comparison.compare import evaluate_export, main as compare_main
from Classifier.data import Images, audit
from Classifier.model import DoRALinear
from Classifier.run import checkpoint, restore_adapter, seed_all, sha256


class ControlledComparisonTests(unittest.TestCase):
    def test_protocol_is_exact_and_seed_is_fixed(self):
        config = Path("Classifier/abc_comparison/config.json")
        expected = {
            "A": ("head_only", "ce", "max_softmax"),
            "B": ("head_only", "l2", "max_sigmoid"),
            "C": ("lora_head", "ce", "max_softmax"),
            "E": ("dora_head", "l2", "max_sigmoid"),
        }
        for group, values in expected.items():
            cfg = resolve_config(config, group)
            self.assertEqual((cfg["tuning_mode"], cfg["loss"], cfg["native_confidence"]), values)
            self.assertEqual((cfg["seed"], cfg["epochs"], cfg["batch_size"], cfg["lr_warmup"]),
                             (42, 20, 32, 2))

    def test_loss_and_prediction_definitions(self):
        logits = torch.tensor([[2.0, 1.0, -1.0], [-2.0, 0.0, 1.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        for group in "ABCE":
            cfg = {**GROUPS[group], "bce_reduction": "mean"}
            selected, ce, l2 = loss_values(logits, labels, cfg)
            self.assertTrue(torch.isfinite(selected))
            self.assertEqual(selected.data_ptr(), (ce if cfg["loss"] == "ce" else l2).data_ptr())
        top, scores = confidence_values(logits)
        self.assertTrue(torch.equal(logits.argmax(1), top.indices[:, 0]))
        self.assertTrue(torch.all((scores["max_softmax"] >= 0) & (scores["max_softmax"] <= 1)))
        self.assertTrue(torch.all((scores["max_sigmoid"] >= 0) & (scores["max_sigmoid"] <= 1)))

    def test_head_initialization_and_trainable_parameters_are_controlled(self):
        base = json.loads(Path("Classifier/abc_comparison/config.json").read_text(encoding="utf-8"))
        base.update(model="vit_tiny_patch16", image_size=32, num_classes=3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_cfg = {**base, **GROUPS["A"], "group": "A"}
            seed_all(42)
            source, _ = build_model(base_cfg, initialize=False)
            state = {
                key.replace("fc_norm.", "norm."): value
                for key, value in source.state_dict().items()
                if not key.startswith("head.")
            }
            state["decoder_embed.weight"] = torch.zeros(2, 2)
            base["pretrained"] = str(root / "mae.pt")
            torch.save({"model": state, "args": Namespace(model="tiny")}, base["pretrained"])

            models, reports = {}, {}
            for group in "ABCE":
                cfg = {**base, **GROUPS[group], "group": group}
                seed_all(42)
                model, report = build_model(cfg)
                models[group] = model
                reports[group] = report
                trainable = [name for name, p in model.named_parameters() if p.requires_grad]
                if group in "AB":
                    self.assertTrue(trainable and all(name.startswith("head.") for name in trainable))
                    self.assertEqual(len(make_optimizer(model, cfg).param_groups), 1)
                else:
                    self.assertTrue(any(".a.weight" in name for name in trainable))
                    self.assertEqual(len(make_optimizer(model, cfg).param_groups), 2)
                    self.assertEqual(len(report["target_modules"]), 12)
                    self.assertEqual(
                        any(name.endswith(".magnitude") for name in trainable), group == "E"
                    )
            self.assertEqual(
                reports["E"]["trainable_parameters"] - reports["C"]["trainable_parameters"],
                12 * (3 * 192),
            )
            torch.testing.assert_close(models["A"].head.weight, models["B"].head.weight, rtol=0, atol=0)
            torch.testing.assert_close(models["A"].head.weight, models["C"].head.weight, rtol=0, atol=0)
            torch.testing.assert_close(models["A"].head.weight, models["E"].head.weight, rtol=0, atol=0)
            for c_block, e_block in zip(models["C"].blocks, models["E"].blocks):
                torch.testing.assert_close(c_block.attn.qkv.a.weight, e_block.attn.qkv.a.weight,
                                           rtol=0, atol=0)
                torch.testing.assert_close(c_block.attn.qkv.b.weight, e_block.attn.qkv.b.weight,
                                           rtol=0, atol=0)
            x = torch.randn(2, 3, 32, 32)
            for model in models.values():
                model.eval()
            with torch.no_grad():
                torch.testing.assert_close(models["A"](x), models["B"](x), rtol=0, atol=0)
                torch.testing.assert_close(models["A"](x), models["C"](x), rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(models["A"](x), models["E"](x), rtol=1e-6, atol=1e-6)

    def test_dora_matches_dense_formula_and_trains_all_adapter_parts(self):
        seed_all(42)
        base = torch.nn.Linear(7, 9, bias=True)
        layer = DoRALinear(base, rank=3, alpha=6, dropout=0.0)
        x = torch.randn(2, 4, 7)
        with torch.no_grad():
            torch.testing.assert_close(layer(x), base(x), rtol=1e-6, atol=1e-6)
            layer.b.weight.normal_(mean=0.0, std=0.02)
            layer.magnitude.mul_(1.03)
            update = layer.b.weight @ layer.a.weight * layer.scale
            direction = base.weight + update
            dense_weight = layer.magnitude[:, None] * direction / direction.norm(dim=1)[:, None]
            expected = torch.nn.functional.linear(x, dense_weight, base.bias)
            torch.testing.assert_close(layer(x), expected, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(layer._weight_norm(), direction.norm(dim=1),
                                       rtol=2e-6, atol=2e-6)
        loss = layer(x).square().mean()
        loss.backward()
        self.assertIsNone(base.weight.grad)
        self.assertIsNotNone(layer.a.weight.grad)
        self.assertIsNotNone(layer.b.weight.grad)
        self.assertIsNotNone(layer.magnitude.grad)
        self.assertTrue(torch.isfinite(layer.magnitude.grad).all())

    def test_tiny_train_checkpoint_and_export(self):
        base = json.loads(Path("Classifier/abc_comparison/config.json").read_text(encoding="utf-8"))
        base.update(model="vit_tiny_patch16", image_size=32, num_classes=2, epochs=2,
                    batch_size=2, workers=0, lr_warmup=0, diagnostic_batches=1,
                    amp=False, **GROUPS["A"], group="A")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = {"a": 0, "b": 1}
            for split in ("train", "val"):
                for name in mapping:
                    folder = root / split / name
                    folder.mkdir(parents=True)
                    for index in range(2):
                        Image.new("RGB", (40, 48), (index * 40, 50, 100)).save(
                            folder / f"{split}_{index}.png"
                        )
            seed_all(42)
            source, _ = build_model(base, initialize=False)
            state = {key.replace("fc_norm.", "norm."): value
                     for key, value in source.state_dict().items() if not key.startswith("head.")}
            state["decoder_embed.weight"] = torch.zeros(2, 2)
            base["pretrained"] = str(root / "mae.pt")
            torch.save({"model": state, "args": Namespace(model="tiny")}, base["pretrained"])
            model, _ = build_model(base)
            train = Images(root / "train", mapping, 32, train=True)
            val = Images(root / "val", mapping, 32)
            stats = audit(train, val)
            optimizer = make_optimizer(model, base)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            metrics = train_epoch(
                model, train, optimizer, scaler, torch.device("cpu"), base, 0, root
            )
            self.assertEqual(metrics["optimizer_steps"], 2)
            digest = sha256(base["pretrained"])
            checkpoint(root / "last.pt", model, optimizer, scaler, base, 0, 0.0,
                       mapping, stats, digest)
            saved = torch.load(root / "last.pt", weights_only=False)
            restored, _ = build_model(base)
            restore_adapter(restored, saved)
            output = root / "export"
            result = evaluate(restored, val, base, torch.device("cpu"), stats["train_counts"], output)
            self.assertEqual(result["samples"], 4)
            self.assertEqual(json.loads((output / "complete.json").read_text())["probability_transform"],
                             "softmax")
            probabilities = np.load(output / "probabilities.npy")
            np.testing.assert_allclose(probabilities.sum(1), 1.0, rtol=1e-5, atol=1e-6)

            # Exercise the complete E path: train, checkpoint, restore and sigmoid export.
            e_cfg = {**base, **GROUPS["E"], "group": "E"}
            e_run = root / "e_run"
            e_run.mkdir()
            seed_all(42)
            e_model, _ = build_model(e_cfg)
            e_optimizer = make_optimizer(e_model, e_cfg)
            e_scaler = torch.amp.GradScaler("cuda", enabled=False)
            e_metrics = train_epoch(
                e_model, train, e_optimizer, e_scaler, torch.device("cpu"), e_cfg, 0, e_run
            )
            self.assertEqual(e_metrics["optimizer_steps"], 2)
            checkpoint(e_run / "last.pt", e_model, e_optimizer, e_scaler, e_cfg, 0, 0.0,
                       mapping, stats, digest)
            e_saved = torch.load(e_run / "last.pt", weights_only=False)
            self.assertTrue(any(name.endswith(".magnitude") for name in e_saved["adapter"]))
            e_restored, _ = build_model(e_cfg)
            restore_adapter(e_restored, e_saved)
            test_input = torch.randn(2, 3, 32, 32)
            e_model.eval()
            e_restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(e_model(test_input), e_restored(test_input),
                                           rtol=0, atol=0)
            e_output = root / "e_export"
            e_result = evaluate(
                e_restored, val, e_cfg, torch.device("cpu"), stats["train_counts"], e_output
            )
            self.assertEqual(e_result["samples"], 4)
            self.assertEqual(
                json.loads((e_output / "complete.json").read_text())["probability_transform"],
                "sigmoid",
            )
            e_probabilities = np.load(e_output / "probabilities.npy")
            self.assertTrue(np.all((e_probabilities >= 0) & (e_probabilities <= 1)))

    def test_compare_reads_existing_d_style_export_and_rejects_order_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            export = run / "val_export"
            export.mkdir(parents=True)
            logits = np.asarray([[2., 0., -1.], [0., 3., 1.], [1., 2., 0.]], dtype=np.float32)
            np.save(export / "logits.npy", logits)
            (export / "complete.json").write_text(
                json.dumps({"rows": 3, "columns": 3, "dtype": "float32"}), encoding="utf-8"
            )
            (run / "data_audit.json").write_text(
                json.dumps({"train_counts": [10, 10, 10]}), encoding="utf-8"
            )
            with (export / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_id", "ground_truth"])
                writer.writerows([("a.jpg", 0), ("b.jpg", 1), ("c.jpg", 2)])
            ids, metrics = evaluate_export(export, chunk_size=2)
            self.assertEqual(ids, ["a.jpg", "b.jpg", "c.jpg"])
            self.assertEqual(set(metrics), {"max_softmax", "max_sigmoid", "sigmoid_logit_margin"})
            self.assertAlmostEqual(metrics["max_softmax"]["top1"], 2/3)
            with self.assertRaises(ValueError):
                evaluate_export(export, expected_ids=["b.jpg", "a.jpg", "c.jpg"], chunk_size=2)
            comparison = Path(temporary) / "comparison"
            argv = ["compare"]
            for group in "abcde":
                argv.extend([f"--{group}", str(export)])
            argv.extend(["--output", str(comparison), "--chunk-size", "2"])
            with mock.patch("sys.argv", argv):
                compare_main()
            contrasts = json.loads((comparison / "contrasts.json").read_text(encoding="utf-8"))
            self.assertIn("E_minus_D_dora_effect_under_l2", contrasts)
            self.assertAlmostEqual(
                contrasts["E_minus_D_dora_effect_under_l2"]["top1"], 0.0
            )
            marker = json.loads((comparison / "complete.json").read_text(encoding="utf-8"))
            self.assertEqual(set(marker["groups"]), set("ABCDE"))


if __name__ == "__main__":
    unittest.main()
