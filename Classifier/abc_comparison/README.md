# A/B/C/D/E：Head-only、LoRA+Head 与 DoRA+Head 的受控对比（seed=42）

本目录实现 A/B/C/E；已有的 `Classifier` L2 run 是 D。源码不再拆子目录。

| Group | Trainable | Loss | Native confidence |
|---|---|---|---|
| A | Head only | Cross-Entropy | Max Softmax |
| B | Head only | L2 (one-vs-rest BCE) | Max Sigmoid |
| C | LoRA + Head | Cross-Entropy | Max Softmax |
| D | LoRA + Head | L2 (existing run) | Max Sigmoid |
| E | DoRA + Head | L2 (one-vs-rest BCE) | Max Sigmoid |

五组的预测均为 `argmax(logits)`；Softmax/Sigmoid 只影响置信度。A/B/C/D/E
共享已确认的数据、PlantCLEF MAE ViT-L/16、224px、纯线性 4271 类新 head、
增强、20 epochs、batch 32、2-epoch LR warmup、AdamW、weight decay 0.05、AMP、
梯度裁剪和 best-by-val-Top1 规则。seed 强制为 42。Head-only 只使用 head
LR=1e-3；C/D/E 的 adapter LR=1e-4、head LR=1e-3。C/D 使用相同 QKV LoRA，
E 只将 D 的 QKV LoRA 换成 DoRA；rank=8、alpha=16、dropout=0 和 target modules
全部不变。DoRA 额外训练每个 QKV 输出通道的 magnitude，这是方法定义带来的参数，
不通过降低 rank 人为抵消。

表中的 L2 是历史实验名，真实实现始终是 one-vs-rest
`binary_cross_entropy_with_logits`，对 batch/classes 全元素取 mean，不是 MSE/Brier。
DoRA 按[原论文](https://arxiv.org/abs/2402.09353)将权重幅值与方向分开学习；动态
方向范数遵循论文建议从梯度图 detach。实现使用与稠密 `W + scale*B@A` 严格等价
的低秩范数恒等式，减少 ViT-L 训练时的临时显存，并由单元测试与稠密参考逐元素核对。

以前的 `MisD/train_vit_classifier_head.py` 使用 BN+Linear、LARS 和不同日程，只作历史参考，不作为这里的 A 组。

## 服务器从拉取到 E 组训练

服务器仓库根目录通常为 `/mnt/hdd8t/Mingle/xyyy/Classifier`，先用 Git 自己确认，
不要在其父目录误运行：

```bash
cd /mnt/hdd8t/Mingle/xyyy/Classifier
git rev-parse --show-toplevel
git status --short --branch
git pull --ff-only origin main

# CUDA torch/torchvision 沿用服务器已验证版本；安装其余固定依赖
python -m pip install -r Classifier/requirements.txt

python -m unittest \
  Classifier.abc_comparison.test_abc \
  Classifier.test_stage1 \
  Classifier.test_official_val \
  Classifier.test_failure_eval -v

test -d /mnt/hdd8t/Mingle/xyyy/MisD/data/classifier_train
test -d /mnt/hdd8t/Mingle/xyyy/MisD/data/classifier_val
test -f /mnt/hdd8t/Mingle/xyyy/MisD/data/class_to_idx.json
test -f /mnt/hdd8t/Mingle/xyyy/models/misclassification-aware/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth
```

若服务器必须使用私有 `config.local.json`，只允许迁移路径和 workers；正式协议字段必须
仍为 20 epochs、batch 32、seed 42、rank 8、alpha 16、dropout 0、adapter/head LR
`1e-4/1e-3`、weight decay 0.05 和 warmup 2。运行时显式传
`--config Classifier/abc_comparison/config.local.json`，并让 smoke、正式训练和导出使用
同一份配置。

## 测试

从仓库根目录执行：

```bash
python -m unittest Classifier.abc_comparison.test_abc -v
```

测试会检查 A/B/C/E 定义、seed/epoch/batch 固定、CE/BCE、冻结范围、优化器
参数组、相同 head/低秩初始化、DoRA magnitude 初始化、DoRA 稠密参考公式，以及
C/E 在训练前均不改变预训练输出。

## 训练 A/B/C/E

每组使用独立目录；目录已存在时程序拒绝覆盖。

服务器先用正式配置、真实数据和真实 PlantCLEF 权重跑两个 train/val batch 的基础设施
smoke。它不会写 `best.pt`/`last.pt`，不能作为实验结果或正式训练的恢复点：

```bash
python -m Classifier.abc_comparison.run smoke --group E \
  --output Classifier/outputs/abcde_seed42/E_smoke \
  --smoke-batches 2

test -f Classifier/outputs/abcde_seed42/E_smoke/smoke.json
test ! -e Classifier/outputs/abcde_seed42/E_smoke/best.pt
```

smoke 通过后，从同一 PlantCLEF 权重重新初始化，在新的正式目录完整训练 20 轮：

```bash
python -m Classifier.abc_comparison.run train --group A \
  --output Classifier/outputs/abc_seed42/A

python -m Classifier.abc_comparison.run train --group B \
  --output Classifier/outputs/abc_seed42/B

python -m Classifier.abc_comparison.run train --group C \
  --output Classifier/outputs/abc_seed42/C

python -m Classifier.abc_comparison.run train --group E \
  --output Classifier/outputs/abcde_seed42/E
```

可以在不同 GPU 并行运行，例如在命令前设置 `CUDA_VISIBLE_DEVICES=0`。同一 GPU 上建议串行。

完整 epoch 后中断可从对应 `last.pt` 恢复，例如：

```bash
python -m Classifier.abc_comparison.run train --group A \
  --output Classifier/outputs/abc_seed42/A \
  --checkpoint Classifier/outputs/abc_seed42/A/last.pt
```

E 的断点恢复命令：

```bash
python -m Classifier.abc_comparison.run train --group E \
  --output Classifier/outputs/abcde_seed42/E \
  --checkpoint Classifier/outputs/abcde_seed42/E/last.pt
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

run=Classifier/outputs/abcde_seed42/E
python -m Classifier.abc_comparison.run export --group E \
  --config "$run/config.json" \
  --checkpoint "$run/best.pt" \
  --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/classifier_val \
  --output "$run/val_export"
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
  --e Classifier/outputs/abcde_seed42/E/val_export \
  --output Classifier/outputs/abcde_seed42/comparison
```

比较器先严格确认五组 sample_id 及顺序完全相同，再从各自 logits 分批重算
Max-Softmax、Max-Sigmoid、sigmoid(logit margin)；输出 `metrics_by_score.csv` 和
`contrasts.json`。新增的主要对比是 `E−D`（BCE/Max-Sigmoid 下 DoRA 相对 LoRA）；
`E−B` 是 DoRA+Head 相对 Head-only 的总增益。原 2×2 对比继续保留。只有一个 seed，
因此结果是描述性对比，不能据此声称统计显著。

模型选择和所有调参只使用 classifier_val。阈值在后续 detector_calibration 上确定；official_val 留作最终一次评估。

## A/B/C/D/E 统一 official_val 评估

五组完成训练后，优先使用统一入口而不是分别手工导出。该入口会先核对五个
`best.pt` 的 seed、训练协议、训练数据审计、预训练权重 SHA256 和类别映射，随后
让五组按完全相同的确定性预处理及样本顺序评估。B/D/E 表中的 L2 会明确记录为
实际实现 `BCEWithLogitsLoss`（one-vs-rest、batch/classes 全元素 mean），不是
MSE/Brier。

先做不覆盖正式目录的 smoke test：

```bash
python -m Classifier.evaluate_official_val \
  --a-run Classifier/outputs/abc_seed42/A \
  --b-run Classifier/outputs/abc_seed42/B \
  --c-run Classifier/outputs/abc_seed42/C \
  --d-run Classifier/outputs/ep20/l2_s42_retry1 \
  --e-run Classifier/outputs/abcde_seed42/E \
  --output Classifier/outputs/official_val_abcde_seed42_smoke \
  --max-samples 64
```

smoke 成功后换一个新目录执行完整 42,710 张评估：

```bash
python -m Classifier.evaluate_official_val \
  --a-run Classifier/outputs/abc_seed42/A \
  --b-run Classifier/outputs/abc_seed42/B \
  --c-run Classifier/outputs/abc_seed42/C \
  --d-run Classifier/outputs/ep20/l2_s42_retry1 \
  --e-run Classifier/outputs/abcde_seed42/E \
  --output Classifier/outputs/official_val_abcde_seed42
```

若 D 的真实成功 run 不是示例中的 `l2_s42_retry1`，必须传真实目录。每组输出
`<GROUP>_samples.csv`，汇总写入 `summary.json`/`summary.csv`；只有最后出现
`complete.json` 才表示五组全部成功。输出目录已存在时程序拒绝覆盖。E 的设计、
rank、学习率或其他超参数不能根据 official_val 结果调整。
