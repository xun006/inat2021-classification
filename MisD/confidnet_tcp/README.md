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

## Classifier C (LoRA + Head, CE) rerun

The controlled Classifier C checkpoint is an adapter checkpoint rather than the
legacy full-model checkpoint. Use the dedicated script; do not pass `best.pt`
to the legacy command. It reconstructs the MAE backbone, restores C's LoRA and
linear head, verifies the checkpoint/config/backbone/class map, and writes to a
separate namespace:

```bash
bash MisD/confidnet_tcp/run_train_classifier_c.sh
```

Default outputs are under
`MisD/output/confidnet_tcp_classifier_c_seed42/`, including a teacher-specific
cache and ConfidNet seeds 0, 1, and 2. Existing `MisD/output/confidnet_tcp/`
results are never overwritten. Override `CLASSIFIER_RUN`, `CLASSIFIER_CONFIG`,
`TEACHER_CHECKPOINT`, or `OUT` as environment variables when paths differ.

After selecting a detector checkpoint using `detector_calibration`, run the
held-out split exactly once:

```bash
DETECTOR_CHECKPOINT=/path/to/selected/checkpoint_best.pth \
  bash MisD/confidnet_tcp/run_final_eval_classifier_c.sh
```

Version-2 TCP caches contain teacher and class-map content identities plus the
ordered relative sample keys. Training rejects stale caches, a different C
checkpoint, or a changed/reordered dataset instead of silently mixing targets.
