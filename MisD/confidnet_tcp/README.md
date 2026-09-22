# Two-stage TCP-ConfiDNet

This is a strict implementation of the two-stage protocol from *Addressing
Failure Prediction by Learning Model Confidence* for the frozen 4,271-class
PlantCLEF ViT classifier.

## Protocol

- The teacher ViT and its classifier head are immutable.
- The detector owns a full independent copy of the teacher's ViT feature
  encoder. It is initialized identically but is never used for classification.
- Stage `head` freezes the copied encoder and trains a confidence MLP to regress
  teacher TCP, `softmax(teacher_logits)[true_label]`.
- Stage `finetune` initializes from Stage `head`, disables dropout, and updates
  only the copied encoder plus confidence MLP. TCP remains a fixed teacher target.
- `detector_train` trains, `detector_calibration` selects/early-stops, and
  `official_val` can only be read by `evaluate.py --allow-official-val`.

The detector score is `1 - confidence`; larger values mean more likely to be a
teacher Top-1 error. The primary selection metric is Error-AUPRC.

## Development run

From `/mnt/hdd8t/Mingle/xyyy`:

```bash
bash MisD/confidnet_tcp/run_train.sh
```

It exports compact target caches for the two development splits, then runs three
seeds through both stages. A cache stores feature/TCP/metadata, not full logits.
Stage B uses cached TCP targets but runs the independent detector encoder on each
image so gradients can specialize its representation.

## Final evaluation

Choose the seed/checkpoint from calibration results first. Then edit the
checkpoint path in `run_final_eval.sh` and run it exactly once:

```bash
bash MisD/confidnet_tcp/run_final_eval.sh
```

The final command writes `predictions.csv` and `metrics.json`, including the
teacher checksum and an explicit `official_val_used: true` audit marker.
