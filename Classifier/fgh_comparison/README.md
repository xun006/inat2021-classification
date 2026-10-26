# F/G/H：预训练初始化与训练范围对照（seed=42）

本目录独立实现三组新增分类器实验，不修改或覆盖既有 A/B/C/D/E 代码与结果。

| 组 | 初始化 | 可训练参数 | Loss | 预测/原生置信度 | 实验含义 |
|---|---|---|---|---|---|
| F | 完全随机 | QKV LoRA + 新 Head | one-vs-rest BCEWithLogits | argmax(logits) / Max-Sigmoid | 严格初始化负对照；不是有效的完整 scratch 训练 |
| G | PlantCLEF checkpoint | 全部 Backbone + 新 Head | one-vs-rest BCEWithLogits | argmax(logits) / Max-Sigmoid | PlantCLEF 初始化下的 Full Fine-tuning |
| H | 完全随机 | 全部 Backbone + 新 Head | one-vs-rest BCEWithLogits | argmax(logits) / Max-Sigmoid | 真正的 from-scratch baseline |

历史文档里的“L2”实际是 `binary_cross_entropy_with_logits`：将单标签转换为
4271 维 one-hot target，并对 batch 和所有类别元素取 mean。它不是 MSE，也不是
Brier loss。本目录在配置兼容字段中仍使用 `loss=l2`，但所有报告都写真实实现名称。

## 固定实验协议

- ViT-Large/16：patch size 16、embedding dim 1024、24 blocks、16 attention heads、
  MLP ratio 4、QKV bias、global average pooling。
- 输入 224×224；分类头为纯 Linear 1024→4271，并在三组中以相同 seed 重新初始化。
- train augmentation 与 D 组一致：RandomResizedCrop(scale 0.5–1.0, bicubic)、
  RandomHorizontalFlip、ImageNet normalization。
- validation/official_val 预处理与 D 组一致：Resize(256)、CenterCrop(224)、
  ImageNet normalization。
- 20 epochs、batch size 32、seed 42、AdamW、weight decay 0.05、gradient clip 1.0、AMP。
- 2-epoch linear warmup 后 cosine decay；Backbone/LoRA LR=1e-4，Head LR=1e-3。
- 每轮在 `classifier_val` 计算指标，只按 Top-1 选择 `best.pt`。
- F 的 LoRA 与 D 相同：每层 attention fused QKV，rank 8、alpha 16、dropout 0。
- `official_val` 不参与训练、超参数选择或 checkpoint 选择。

本实验保持“名义 batch size=32”和单进程单 GPU。若 Full ViT-L 在服务器上显存不足，
不要静默减小 batch 或改变梯度累积；先停止并记录显存情况，再决定是否为所有需要比较的
全量训练组统一增加梯度累积协议。

## 目录和输出

建议正式输出：

```text
Classifier/outputs/fgh_seed42/
├── F/
├── G/
├── H/
├── F_smoke/
├── G_smoke/
└── H_smoke/
```

每个正式 run 包含：

- `config.json`：解析后的冻结配置；
- `data_audit.json`、`class_to_idx.json`：数据指纹和标签映射；
- `model_report.json`：结构、初始化身份、总参数/可训练参数及软件版本；
- `history.jsonl`：每轮 train/validation 指标；
- `diagnostics.jsonl`、`amp_events.jsonl`：早期梯度诊断和 AMP overflow；
- `last.pt`：模型、optimizer、scaler 和 RNG，供精确断点恢复；
- `best.pt`：最佳 epoch 的模型状态和实验元数据，不保存 optimizer；
- 导出目录中的 `samples.csv`、`metrics.json`、`logits.npy`、`probabilities.npy`、
  `per_class.csv`、`risk_coverage.npz`、`provenance.json` 和 `complete.json`。

G/H 为全量 ViT-L，checkpoint 明显大于 F。42,710×4,271 的 float32 logits 或概率
矩阵每个约 0.73 GB；同时保存两者时每组导出约需 1.46 GB，另加 CSV 和指标文件。
所有 `Classifier/outputs/` 内容已由 `Classifier/.gitignore` 排除，不应提交。

## 服务器：从 git pull 到测试

先确认实际仓库根目录，不要依据父目录名称猜测：

```bash
cd /mnt/hdd8t/Mingle/xyyy/Classifier
git rev-parse --show-toplevel
git status --short --branch
git pull --ff-only origin main

python -m pip install -r Classifier/requirements.txt
python -m unittest \
  Classifier.fgh_comparison.test_fgh \
  Classifier.abc_comparison.test_abc \
  Classifier.test_stage1 -v

test -d /mnt/hdd8t/Mingle/xyyy/MisD/data/classifier_train
test -d /mnt/hdd8t/Mingle/xyyy/MisD/data/classifier_val
test -d /mnt/hdd8t/Mingle/xyyy/MisD/data/official_val
test -f /mnt/hdd8t/Mingle/xyyy/MisD/data/class_to_idx.json
test -f /mnt/hdd8t/Mingle/xyyy/models/misclassification-aware/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth
```

程序会拒绝非连续或不匹配的标签映射，并核对正式 train/validation 样本数为
956,507/42,710。只有 G 会读取 PlantCLEF checkpoint；F/H 不会把任何预训练权重
载入模型。F 还会保存随机冻结 backbone 的 SHA256 指纹，评估时重建不一致会直接报错。

若服务器路径不同，可复制为不提交的
`Classifier/fgh_comparison/config.local.json`，只修改路径和 `workers`。模型和训练协议字段
被代码锁定，不能通过本地配置悄悄改变。

## 真实数据 smoke test

每组先运行两个训练 batch 和两个 validation batch。smoke 不生成 `best.pt/last.pt`，
不能作为正式结果或恢复点；输出目录必须预先不存在。

```bash
for group in F G H; do
  python -m Classifier.fgh_comparison.run smoke --group "$group" \
    --output "Classifier/outputs/fgh_seed42/${group}_smoke" \
    --smoke-batches 2
done

for group in F G H; do
  test -f "Classifier/outputs/fgh_seed42/${group}_smoke/smoke.json"
  test ! -e "Classifier/outputs/fgh_seed42/${group}_smoke/best.pt"
done
```

分别检查三个 `smoke.json`：`status=complete`、loss/logits 为有限值、
`optimizer_steps>0`，且没有连续 AMP overflow。

## 正式训练

以下命令各使用一个 GPU，可串行执行，也可通过不同 `CUDA_VISIBLE_DEVICES` 在三张 GPU
上并行。不要在同一个输出目录启动两个进程。

```bash
python -m Classifier.fgh_comparison.run train --group F \
  --output Classifier/outputs/fgh_seed42/F

python -m Classifier.fgh_comparison.run train --group G \
  --output Classifier/outputs/fgh_seed42/G

python -m Classifier.fgh_comparison.run train --group H \
  --output Classifier/outputs/fgh_seed42/H
```

完整 epoch 后中断，只能从对应 run 的 `last.pt` 恢复，例如：

```bash
python -m Classifier.fgh_comparison.run train --group H \
  --config Classifier/outputs/fgh_seed42/H/config.json \
  --output Classifier/outputs/fgh_seed42/H \
  --checkpoint Classifier/outputs/fgh_seed42/H/last.pt
```

## classifier_val 导出与比较

```bash
for group in F G H; do
  run="Classifier/outputs/fgh_seed42/${group}"
  python -m Classifier.fgh_comparison.evaluate \
    --run "$run" \
    --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/classifier_val \
    --expected-samples 42710 \
    --output "$run/val_export"
done

python -m Classifier.fgh_comparison.compare \
  --f Classifier/outputs/fgh_seed42/F/val_export \
  --g Classifier/outputs/fgh_seed42/G/val_export \
  --h Classifier/outputs/fgh_seed42/H/val_export \
  --d Classifier/outputs/ep20/l2_s42_retry1/val_export \
  --output Classifier/outputs/fgh_seed42/comparison
```

D 的目录只是历史示例，必须换成服务器上真实、协议匹配的 D 导出路径。比较器要求
所有组的 `sample_id`、真实标签和顺序逐项相同。核心对比为：

- `G−H`：Full Training 下 PlantCLEF 初始化相对随机初始化的收益；
- `H−F`：随机初始化下，全量训练相对冻结随机 backbone+LoRA 的收益；
- `F−D`（提供 D 时）：LoRA 条件下随机初始化相对 PlantCLEF 初始化的差异；
- `G−D`（提供 D 时）：PlantCLEF 初始化下 Full Fine-tuning 相对 LoRA 的差异。

## official_val：先 smoke，再完整评估

模型选择全部冻结后，每组先评估相同的 64 张前缀样本：

```bash
for group in F G H; do
  run="Classifier/outputs/fgh_seed42/${group}"
  python -m Classifier.fgh_comparison.evaluate \
    --run "$run" \
    --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/official_val \
    --expected-samples 42710 \
    --max-samples 64 \
    --output "$run/official_val_smoke"
done
```

smoke 全部完成后换新目录运行完整 official_val：

```bash
for group in F G H; do
  run="Classifier/outputs/fgh_seed42/${group}"
  python -m Classifier.fgh_comparison.evaluate \
    --run "$run" \
    --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/official_val \
    --expected-samples 42710 \
    --output "$run/official_val_export"
done

python -m Classifier.fgh_comparison.compare \
  --f Classifier/outputs/fgh_seed42/F/official_val_export \
  --g Classifier/outputs/fgh_seed42/G/official_val_export \
  --h Classifier/outputs/fgh_seed42/H/official_val_export \
  --output Classifier/outputs/fgh_seed42/official_val_comparison
```

## 成功判据

正式训练成功必须同时满足：

1. `history.jsonl` 恰好 20 行，epoch 为 0–19；
2. 每组存在 `best.pt` 和 `last.pt`；
3. `model_report.json` 中 F=`random/lora_head`、G=`plantclef/full`、H=`random/full`；
4. F 只有 LoRA+Head 可训练，G/H 的 trainable fraction 为 1；
5. 三组数据指纹和 `class_to_idx.json` 完全一致；
6. validation loss、Top-1 和 logits 均有限，没有持续 AMP overflow；
7. 导出目录存在 `complete.json`，其 rows=42,710；
8. 比较目录存在 `complete.json`，且三组样本身份和顺序检查通过。

当前仅有 seed=42，结果只能作为单 seed 描述性比较，不能报告均值、标准差或显著性。
分类准确率更高也不能直接证明错分检测性能更好；后续仍需在相同 official_val 样本上统一
比较 AUROC、AUPR-Error、AURC/E-AURC 和 FPR@95TPR。
