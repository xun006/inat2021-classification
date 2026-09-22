# Patch Cross-Attention: new-data Stage 1

All paths, scripts, metadata and outputs in this stage are under `MisD`.

Fixed protocol:

- teacher: `MisD/output/vit_large_linear_probe_4271/checkpoint_best.pth`
- fit: `MisD/data/detector_train`
- selection/calibration: `MisD/data/detector_calibration`
- final test: `MisD/data/official_val`
- patch tokens: extracted online and never stored

Run the GPU stage from the project root:

```bash
bash MisD/patch_cross_attention/run_gpu_stage_1.sh
```

The script exports lightweight prediction CSVs, evaluates training-free B0
confidence baselines on `official_val`, and runs the 100-sample-per-split Gate-0
online consistency test. Do not start detector training unless Gate 0 passes.

## Patch model validation and seed-0 ablation

CPU architecture tests:

```bash
cd MisD/patch_cross_attention
python test_models.py
```

Run the real-image GPU sanity check first (256 balanced training images, 512
calibration images, M2 only, and no access to `official_val`):

```bash
bash MisD/patch_cross_attention/run_gpu_sanity.sh
```

Inspect `MisD/output/patch_cross_attention/sanity/m2_real_images`. Only after
the sanity result is accepted, run the full seed-0 ablation:

```bash
bash MisD/patch_cross_attention/run_gpu_seed0_ablation.sh
```
