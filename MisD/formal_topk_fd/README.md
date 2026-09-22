# Formal Top-K Competition-Aware Failure Detector

Implementation of the locked main method in `MisD/植物细粒度错误分类检测实验任务书.md`.

The training program opens only `detector_train` and `detector_calibration`.
It never constructs an `official_val` dataset. The frozen teacher always owns
the class prediction; this detector only outputs the probability that the
teacher Top-1 prediction is wrong.

## CPU architecture tests

```bash
cd /mnt/hdd8t/Mingle/xyyy
python MisD/formal_topk_fd/test_formal_method.py
```

## Seed-0 training (GPU)

```bash
cd /mnt/hdd8t/Mingle/xyyy
bash MisD/formal_topk_fd/run_seed0.sh
```

The selected checkpoint and calibration outputs are written to
`MisD/output/formal_topk_fd/seed0`. Selection uses calibration Error AUPRC.

## Overfitting-improvement screen

The first run peaked at epoch 3 and then overfit. The controlled screen tests,
in order: compressed positive weight, removal of pair loss, a smaller matching
network, and a direct correctness head. Each run changes only the factors
documented in `run_improvement_screen.sh` and never reads `official_val`.

```bash
cd /mnt/hdd8t/Mingle/xyyy
bash MisD/formal_topk_fd/run_improvement_screen.sh
python MisD/formal_topk_fd/summarize_screen.py
```

The already completed 7-D probability-shape MLP remains the probability-only
reference (calibration Error AUPRC 0.57504). It is not retrained inside this GPU
screen.

Do not evaluate `official_val` until the calibration result has passed the
predeclared gate and the architecture is locked. At that point run:

```bash
python MisD/formal_topk_fd/evaluate.py \
  --detector-checkpoint MisD/output/formal_topk_fd/seed0/checkpoint_best.pth \
  --split official_val \
  --device cuda
```

The current screening selected `s4_direct` by calibration Error AUPRC. Its
selection record was frozen in `selected_model.json` before official testing.
Run the locked test and produce the common-baseline comparison with:

```bash
cd /mnt/hdd8t/Mingle/xyyy
bash MisD/formal_topk_fd/run_official_selected.sh
```
