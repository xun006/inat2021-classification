"""CPU tests for the isolated F/G/H controlled experiment."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch

from Classifier.fgh_comparison.checkpoint_utils import (
    load as load_checkpoint,
    restore_model,
    save_best,
)
from Classifier.fgh_comparison.compare import main as compare_main
from Classifier.fgh_comparison.model import build_model
from Classifier.fgh_comparison.run import (
    GROUPS,
    frozen_parameter_fingerprint,
    make_optimizer,
    resolve_config,
)
from Classifier.abc_comparison.run import train_epoch
from Classifier.run import seed_all


class FGHComparisonTests(unittest.TestCase):
    def tiny_config(self, group, root):
        base = json.loads(
            Path("Classifier/fgh_comparison/config.json").read_text(encoding="utf-8")
        )
        base.update(
            model="vit_tiny_patch16",
            image_size=32,
            num_classes=3,
            epochs=2,
            batch_size=2,
            workers=0,
            lr_warmup=0,
            amp=False,
            expected_train_samples=None,
            expected_val_samples=None,
            pretrained=str(root / "plantclef.pt"),
            **GROUPS[group],
            group=group,
        )
        return base

    def make_pretrained(self, root):
        cfg = self.tiny_config("H", root)
        seed_all(42)
        source, _ = build_model(cfg)
        state = {
            key.replace("fc_norm.", "norm."): value
            for key, value in source.state_dict().items()
            if not key.startswith("head.")
        }
        state["decoder_embed.weight"] = torch.zeros(2, 2)
        torch.save({"model": state, "args": Namespace(model="tiny")}, root / "plantclef.pt")

    def test_formal_protocol_is_fixed(self):
        path = Path("Classifier/fgh_comparison/config.json")
        expected = {
            "F": ("random", "lora_head"),
            "G": ("plantclef", "full"),
            "H": ("random", "full"),
        }
        for group, identity in expected.items():
            cfg = resolve_config(path, group)
            self.assertEqual((cfg["initialization"], cfg["tuning_mode"]), identity)
            self.assertEqual(cfg["loss"], "l2")
            self.assertEqual(cfg["native_confidence"], "max_sigmoid")
            self.assertEqual(
                (cfg["seed"], cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["head_lr"]),
                (42, 20, 32, 1e-4, 1e-3),
            )

    def test_initialization_trainability_and_optimizer_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_pretrained(root)
            models, reports = {}, {}
            for group in "FGH":
                cfg = self.tiny_config(group, root)
                seed_all(42)
                models[group], reports[group] = build_model(cfg)
                optimizer = make_optimizer(models[group], cfg)
                self.assertEqual([entry["name"] for entry in optimizer.param_groups], [
                    "lora" if group == "F" else "backbone", "head"
                ])

            f_trainable = [
                name for name, parameter in models["F"].named_parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(any(".a.weight" in name for name in f_trainable))
            self.assertTrue(all(
                name.startswith("head.") or ".a.weight" in name or ".b.weight" in name
                for name in f_trainable
            ))
            self.assertEqual(len(reports["F"]["target_modules"]), 12)
            self.assertEqual(
                reports["G"]["trainable_parameters"], reports["G"]["total_parameters"]
            )
            self.assertEqual(
                reports["H"]["trainable_parameters"], reports["H"]["total_parameters"]
            )
            torch.testing.assert_close(
                models["G"].head.weight, models["H"].head.weight, rtol=0, atol=0
            )
            torch.testing.assert_close(
                models["G"].head.bias, models["H"].head.bias, rtol=0, atol=0
            )

            # Zero-initialized LoRA means F and H have identical random-init logits at step zero.
            sample = torch.randn(2, 3, 32, 32)
            models["F"].eval()
            models["H"].eval()
            with torch.no_grad():
                torch.testing.assert_close(models["F"](sample), models["H"](sample))
            self.assertNotEqual(frozen_parameter_fingerprint(models["F"]), "0" * 64)

    def test_peft_and_full_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_pretrained(root)
            sample = torch.randn(2, 3, 32, 32)
            mapping = {"a": 0, "b": 1, "c": 2}
            stats = {"train_counts": [2, 2, 2], "train_samples": 6, "val_samples": 3}
            for group in "FH":
                cfg = self.tiny_config(group, root)
                seed_all(42)
                model, _ = build_model(cfg)
                model.train()
                model(sample).sum().backward()
                with torch.no_grad():
                    for parameter in model.parameters():
                        if parameter.requires_grad and parameter.grad is not None:
                            parameter.add_(parameter.grad, alpha=-1e-4)
                model.eval()
                expected = model(sample).detach()
                path = root / f"{group}.pt"
                save_best(path, model, cfg, 0, 0.5, mapping, stats, f"identity-{group}")
                saved = load_checkpoint(path)
                seed_all(42)
                restored, _ = build_model(cfg)
                restore_model(restored, saved)
                restored.eval()
                torch.testing.assert_close(restored(sample), expected)

    def test_one_real_optimizer_step_for_each_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_pretrained(root)
            dataset = [
                (torch.randn(3, 32, 32), torch.tensor(index % 3), f"{index}.png")
                for index in range(2)
            ]
            for group in "FGH":
                output = root / f"smoke_{group}"
                output.mkdir()
                cfg = self.tiny_config(group, root)
                cfg["diagnostic_batches"] = 0
                seed_all(42)
                model, _ = build_model(cfg)
                optimizer = make_optimizer(model, cfg)
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                metrics = train_epoch(
                    model, dataset, optimizer, scaler, torch.device("cpu"), cfg,
                    epoch=0, output=output, max_batches=1,
                )
                self.assertEqual(metrics["optimizer_steps"], 1)
                self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_compare_rejects_mismatched_samples_and_writes_contrasts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def make_export(group, top1, reverse=False):
                export = root / group
                export.mkdir()
                rows = [("a.jpg", 0, 1), ("b.jpg", 1, 0)]
                if reverse:
                    rows.reverse()
                with (export / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["sample_id", "ground_truth", "correct"])
                    writer.writerows(rows)
                (export / "complete.json").write_text(
                    json.dumps({"rows": 2}), encoding="utf-8"
                )
                (export / "metrics.json").write_text(json.dumps({
                    "samples": 2, "top1": top1, "top5": 1.0, "loss": 0.2,
                }), encoding="utf-8")
                return export

            exports = {
                "f": make_export("F", 0.4),
                "g": make_export("G", 0.8),
                "h": make_export("H", 0.6),
            }
            output = root / "comparison"
            argv = ["compare"]
            for group, path in exports.items():
                argv.extend([f"--{group}", str(path)])
            argv.extend(["--output", str(output)])
            with mock.patch("sys.argv", argv):
                compare_main()
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                summary["contrasts"]["G_minus_H_pretraining_effect_under_full_training"]["top1"],
                0.2,
            )

            bad = make_export("bad", 0.5, reverse=True)
            with self.assertRaises(ValueError):
                from Classifier.fgh_comparison.compare import read_export
                read_export(bad, [("a.jpg", 0), ("b.jpg", 1)])


if __name__ == "__main__":
    unittest.main()
