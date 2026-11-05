# 正式分类器实验协议

## 实验矩阵

| Group | Initialization | Objective | Trainable scope | Epochs | Warmup |
|---|---|---|---|---:|---:|
| A | PlantCLEF MAE | CE | Head only | 20 | 2 |
| B | PlantCLEF MAE | BCE | Head only | 20 | 2 |
| C | PlantCLEF MAE | CE | QKV LoRA + Head | 20 | 2 |
| D | PlantCLEF MAE | BCE | QKV LoRA + Head | 20 | 2 |
| G | PlantCLEF MAE | BCE | Full | 30 | 3 |
| H | Random | BCE | Full | 30 | 3 |
| I | ImageNet MAE | BCE | Full | 30 | 3 |

正式 seeds 为 42、43、44。E、F 已删除。

## 公平性边界

- A–D 保留历史 seed=42 的模型结构、初始化、数据处理、loss、学习率、batch size、
  optimizer、weight decay、20 epochs 和 2 epoch warmup，不改变比较含义。
- G/H/I 使用相同的 Full-training 训练协议：30 epochs、3 epoch warmup。3/30 与 A–D
  的 2/20 都是 10% warmup；其余优化参数在三组中完全相同。
- G/H 的历史 20-epoch 结果不能与新 30-epoch 结果混合。G/H/I 必须重新运行
  seed 42/43/44，才能做初始化对照。
- BCE 保留历史定义：单标签转 4271 维 one-hot，使用
  `binary_cross_entropy_with_logits` 并对 batch/class elements 取 mean。
- 模型只按 `classifier_val` Top-1 选择 `best.pt`；`official_val` 不参与调参或选模。

共同参数：batch size 32、AdamW、weight decay 0.05、gradient clip 1.0、AMP、
backbone/LoRA LR 1e-4、head LR 1e-3、cosine decay。训练范围不同导致的 optimizer
parameter groups 是实验处理的一部分，不允许针对某一 group 看结果后单独改 LR。

## 输出

```text
/mnt/hdd8t/Mingle/xyyy/Classifier/outputs/<GROUP>/seed_<SEED>/
├── amp_events.jsonl
├── best.pt
├── class_to_idx.json
├── config.json
├── data_audit.json
├── diagnostics.jsonl
├── history.jsonl
├── last.pt
└── model_report.json
```

## 服务器预检

```bash
cd /mnt/hdd8t/Mingle/xyyy/Classifier
python -m pip install -r Classifier/requirements.txt

test -f /mnt/hdd8t/Mingle/xyyy/models/PlantCLEF_Model/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth
test -f /mnt/hdd8t/Mingle/xyyy/models/PlantCLEF_Model/mae_pretrain_vit_large.pth

python -m unittest \
  Classifier.test_protocol \
  Classifier.test_stage1 \
  Classifier.test_failure_eval -v
```

每组先跑独立 smoke，smoke 不产生正式 checkpoint：

```bash
python -m Classifier.experiment smoke --group A --seed 42 --batches 2
python -m Classifier.experiment smoke --group I --seed 42 --batches 2
```

## 正式训练与恢复

```bash
python -m Classifier.experiment train --group A --seed 43
python -m Classifier.experiment train --group I --seed 44
python -m Classifier.experiment train --group I --seed 44 --resume
```

按 seed 串行运行全部组：

```bash
bash Classifier/scripts/train_seed.sh 43
bash Classifier/scripts/train_seed.sh 44
```

## 导出、审计和汇总

```bash
python -m Classifier.experiment export --group I --seed 44 \
  --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/classifier_val \
  --name val_export --expected-samples 42710

python -m Classifier.audit_runs \
  --output-root /mnt/hdd8t/Mingle/xyyy/Classifier/outputs \
  --report /mnt/hdd8t/Mingle/xyyy/Classifier/outputs/audit_report.json

python -m Classifier.summarize_matrix \
  --output-root /mnt/hdd8t/Mingle/xyyy/Classifier/outputs \
  --output /mnt/hdd8t/Mingle/xyyy/Classifier/outputs/summary_val
```

汇总器要求同组三个 seed 的配置一致，并要求全部组的数据审计一致；否则拒绝生成结果。
