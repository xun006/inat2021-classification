"""Run: python -m unittest Classifier.test_stage1 -v (CPU, no real weights/data)."""
import json
import random
from argparse import Namespace
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .losses import classification_margin_loss, sigmoid_bce_loss, confidence_ranking_loss, ranking_weight


class LossTests(unittest.TestCase):
    def test_hinge_and_gradient(self):
        self.assertEqual(classification_margin_loss(torch.tensor([[3., 0.]]), torch.tensor([0])).item(), 0.)
        logits = torch.tensor([[0., 3.]], requires_grad=True)
        loss = classification_margin_loss(logits, torch.tensor([0]))
        self.assertEqual(loss.item(), 4.)
        loss.backward()
        self.assertLess(logits.grad[0, 0], 0)
        self.assertGreater(logits.grad[0, 1], 0)

    def test_bce_reduction(self):
        logits = torch.zeros(2, 4271, requires_grad=True)
        labels = torch.tensor([0, 4270])
        mean = sigmoid_bce_loss(logits, labels)
        self.assertAlmostEqual(mean.item(), np.log(2), places=5)
        self.assertAlmostEqual((sigmoid_bce_loss(logits, labels, "class_sum") / mean).item(), 4271, places=2)

    def test_ranking(self):
        labels = torch.tensor([0, 0])
        good = torch.tensor([[3., 0.], [-1., 0.]])
        bad = torch.tensor([[0., -1.], [0., 3.]], requires_grad=True)
        self.assertEqual(confidence_ranking_loss(good, labels)[0].item(), 0.)
        loss, pairs = confidence_ranking_loss(bad, labels)
        self.assertEqual(pairs, 1)
        self.assertGreater(loss.item(), .1)
        loss.backward()
        self.assertLess(bad.grad[0, 0], 0)
        self.assertGreater(bad.grad[1, 1], 0)
        for labels in (torch.tensor([0, 1]), torch.tensor([1, 0])):
            z = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
            loss, pairs = confidence_ranking_loss(z, labels)
            self.assertEqual(pairs, 0)
            loss.backward()
            self.assertTrue(torch.isfinite(z.grad).all())

    def test_all_losses_optimizer_step(self):
        for name in ("l1", "l2", "l1_l3", "l2_l3"):
            head = torch.nn.Linear(4, 3)
            opt = torch.optim.AdamW(head.parameters())
            z = head(torch.randn(8, 4))
            y = torch.arange(8) % 3
            loss = classification_margin_loss(z, y) if name.startswith("l1") else sigmoid_bce_loss(z, y)
            if name.endswith("_l3"):
                loss = loss + confidence_ranking_loss(z, y)[0]
            loss.backward()
            opt.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in head.parameters()))

    def test_schedule(self):
        cfg = dict(loss="l1_l3", lambda3=1., l3_warmup=3, l3_ramp=3)
        self.assertEqual(ranking_weight(cfg, 2), 0)
        self.assertAlmostEqual(ranking_weight(cfg, 3), 1/3)
        self.assertEqual(ranking_weight(cfg, 5), 1)


class PipelineTests(unittest.TestCase):
    def test_metric_ties(self):
        from .metrics import evaluate_arrays
        a = evaluate_arrays([0, 0], [0, 1], [.5, .5], [True, True], [10, 10])[0]
        b = evaluate_arrays([0, 0], [1, 0], [.5, .5], [True, True], [10, 10])[0]
        self.assertEqual(a["aurc"], b["aurc"])
        self.assertEqual(a["auroc_error"], .5)
        self.assertEqual(a["aurc"], .5)

    def test_pipeline_and_reference_compatibility(self):
        from .data import Images, audit
        from .model import build_model
        from .run import train_epoch, evaluate, checkpoint, restore_adapter, seed_all, sha256
        torch.set_num_threads(2)
        cfg = json.loads(Path("Classifier/config.json").read_text())
        cfg.update(model="vit_tiny_patch16", image_size=32, num_classes=2, epochs=2,
                   workers=0, batch_size=2, lr_warmup=0, l3_warmup=0,
                   diagnostic_batches=1, amp=False, loss="l1_l3")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = {"a": 0, "b": 1}
            for split in ("train", "val"):
                for name in mapping:
                    folder = root / split / name
                    folder.mkdir(parents=True)
                    for i in range(2):
                        Image.new("RGB", (40, 48), (i*40, 50, 100)).save(folder / f"{split}_{i}.png")
            # Simulate original MAE encoder state (norm, not fc_norm) plus decoder.
            seed_all(42)
            original, _ = build_model(cfg, initialize=False)
            state = {}
            for k, value in original.state_dict().items():
                if ".a." in k or ".b." in k:
                    continue
                state[k.replace(".base.", ".").replace("fc_norm.", "norm.")] = value
            state["decoder_embed.weight"] = torch.zeros(2, 2)
            cfg["pretrained"] = str(root / "pretrained.pt")
            torch.save({"model": state, "args": Namespace(model="vit_tiny_patch16", epochs=100)}, cfg["pretrained"])
            seed_all(42)
            model, report = build_model(cfg)
            self.assertGreater(report["trainable_parameters"], 0)
            self.assertTrue(all((p.requires_grad == (".a." in k or ".b." in k or k.startswith("head.")))
                                for k, p in model.named_parameters()))
            # Reference PlantCLEF forward: average unnormalized patches then fc_norm.
            model.eval()
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                tokens = model.patch_embed(x)
                tokens = torch.cat((model.cls_token.expand(2, -1, -1), tokens), dim=1) + model.pos_embed
                for block in model.blocks:
                    tokens = block(tokens)
                expected = model.head(model.fc_norm(tokens[:, 1:].mean(1)))
                torch.testing.assert_close(model(x), expected)
            train = Images(root / "train", mapping, 32, True)
            val = Images(root / "val", mapping, 32)
            stats = audit(train, val)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            opt.param_groups[0]["initial_lr"] = .001
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            train_epoch(model, train, opt, scaler, torch.device("cpu"), cfg, 0, root)
            checksum = sha256(cfg["pretrained"])
            checkpoint(root / "last.pt", model, opt, scaler, cfg, 0, 0., mapping, stats, checksum)
            saved = torch.load(root / "last.pt", weights_only=False)
            restored, _ = build_model(cfg)
            restore_adapter(restored, saved)
            model.eval()
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
            # A resumed next epoch must match uninterrupted CPU training exactly.
            def restore_rng():
                torch.set_rng_state(saved["torch_rng"])
                np.random.set_state(saved["numpy_rng"])
                random.setstate(saved["python_rng"])
            restore_rng()
            train_epoch(model, train, opt, scaler, torch.device("cpu"), cfg, 1, root)
            opt2 = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.001)
            opt2.load_state_dict(saved["optimizer"])
            scaler2 = torch.amp.GradScaler("cuda", enabled=False)
            scaler2.load_state_dict(saved["scaler"])
            restore_rng()
            train_epoch(restored, train, opt2, scaler2, torch.device("cpu"), cfg, 1, root)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            result = evaluate(restored, val, cfg, torch.device("cpu"), stats["train_counts"], 0, root / "export")
            self.assertEqual(result["samples"], 4)
            z = np.load(root / "export/logits.npy")
            p = np.load(root / "export/sigmoid.npy")
            self.assertEqual(z.shape, (4, 2))
            np.testing.assert_allclose(p, 1/(1+np.exp(-z)), rtol=1e-6)
            with self.assertRaises(ValueError):
                audit(train, train)


if __name__ == "__main__":
    unittest.main()
