# A/B/C：Head-only 与 LoRA+Head 的受控对比（seed=42）

本目录只实现新增的 A/B/C；已有的 `Classifier` L2 run 是 D。源码不再拆子目录。

| Group | Trainable | Loss | Native confidence |
|---|---|---|---|
| A | Head only | Cross-Entropy | Max Softmax |
| B | Head only | L2 (one-vs-rest BCE) | Max Sigmoid |
| C | LoRA + Head | Cross-Entropy | Max Softmax |
| D | LoRA + Head | L2 (existing run) | Max Sigmoid |

四组的预测均为 `argmax(logits)`；Softmax/Sigmoid 只影响置信度。A/B/C 与 D 共享已确认的数据、PlantCLEF MAE ViT-L/16、224px、纯线性 4271 类新 head、增强、20 epochs、batch 32、2-epoch LR warmup、AdamW、weight decay 0.05、AMP、梯度裁剪和 best-by-val-Top1 规则。seed 强制为 42。Head-only 只使用 head LR=1e-3；C 与 D 额外使用相同 QKV LoRA（rank=8, alpha=16, dropout=0）及 LR=1e-4。

以前的 `MisD/train_vit_classifier_head.py` 使用 BN+Linear、LARS 和不同日程，只作历史参考，不作为这里的 A 组。

## 测试

从仓库根目录执行：

```bash
python -m unittest Classifier.abc_comparison.test_abc -v
```

测试会检查 A/B/C 定义、seed/epoch/batch 固定、CE/L2、冻结范围、优化器参数组、相同 head 初始化，以及 C 的零初始化 LoRA 在训练前不改变输出。

## 训练 A/B/C

每组使用独立目录；目录已存在时程序拒绝覆盖。

```bash
python -m Classifier.abc_comparison.run train --group A \
  --output Classifier/outputs/abc_seed42/A

python -m Classifier.abc_comparison.run train --group B \
  --output Classifier/outputs/abc_seed42/B

python -m Classifier.abc_comparison.run train --group C \
  --output Classifier/outputs/abc_seed42/C
```

可以在不同 GPU 并行运行，例如在命令前设置 `CUDA_VISIBLE_DEVICES=0`。同一 GPU 上建议串行。

完整 epoch 后中断可从对应 `last.pt` 恢复，例如：

```bash
python -m Classifier.abc_comparison.run train --group A \
  --output Classifier/outputs/abc_seed42/A \
  --checkpoint Classifier/outputs/abc_seed42/A/last.pt
```

## 导出 classifier_val

```bash
for group in A B C; do
  run="Classifier/outputs/abc_seed42/${group}"
  python -m Classifier.abc_comparison.run export --group "$group" \
    --config "$run/config.json" \
    --checkpoint "$run/best.pt" \
    --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/classifier_val \
    --output "$run/val_export"
done
```

每组导出 `logits.npy`、原生变换后的 `probabilities.npy`、逐样本 CSV、逐类指标和三种统一置信度的错分指标。CE 的 probabilities 是 Softmax，L2 是 Sigmoid；以 `complete.json` 为准。

## 与 D 统一比较

D 必须是同一数据指纹、预训练 SHA256、20 epochs、batch 32、seed 42、相同 LoRA 配置的 `--loss l2` run，并先导出验证集 logits。假设 D 的导出目录为 `Classifier/outputs/ep20/l2_s42_retry1/val_export`：

```bash
python -m Classifier.abc_comparison.compare \
  --a Classifier/outputs/abc_seed42/A/val_export \
  --b Classifier/outputs/abc_seed42/B/val_export \
  --c Classifier/outputs/abc_seed42/C/val_export \
  --d Classifier/outputs/ep20/l2_s42_retry1/val_export \
  --output Classifier/outputs/abc_seed42/comparison
```

比较器先严格确认四组 sample_id 及顺序完全相同，再从各自 logits 分批重算 Max-Softmax、Max-Sigmoid、sigmoid(logit margin)；输出 `metrics_by_score.csv` 和 `contrasts.json`。核心分类对比为 D−B、C−A、B−A、D−C，以及交互效应 `(D−B)−(C−A)`。只有一个 seed，因此结果是描述性对比，不能据此声称统计显著。

模型选择和所有调参只使用 classifier_val。阈值在后续 detector_calibration 上确定；official_val 留作最终一次评估。
