# 7-D probability-shape MLP

This detector predicts whether the frozen teacher's Top-1 result is wrong.
It uses the seven probability-shape features specified in
`train_probability_shape_mlp.py` and trains from `train_natural.csv`.

Run from any directory:

```bash
python /mnt/hdd8t/Mingle/xyyy/MisD/train_probability_shape_mlp.py
```

By default, outputs are written to `MisD/output/probability_shape_mlp`.
