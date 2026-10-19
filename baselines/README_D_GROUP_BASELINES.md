# D 组模型：无需额外训练的错分检测基线

## 1. 实验目的与范围

本实验固定使用 A/B/C/D 受控对比中的 **D 组最佳模型**：LoRA+Head、one-vs-rest `BCEWithLogitsLoss`（项目中历史名称为 L2）、seed=42，并使用 `classifier_val` Top-1 选择的 `best.pt`。脚本不更新模型权重，也不拟合检测器；七种分数均由同一次 FP32 前向产生的 logits 直接计算。

错误预测为正类：

```text
prediction = argmax(logits)
error_label = 1[prediction != ground_truth]
```

所有 `*_error_score` 均为数值越大越可能错分。

## 2. 七种基线定义

记 (z\in\mathbb{R}^K) 为 logits，(z_{(1)}\ge z_{(2)})，(s_i=\sigma(z_i))，(p=\operatorname{softmax}(z))，预测类别为 \(\hat y=\arg\max_i z_i\)。

| 方法 | 本实现的错分分数 | 含义 |
|---|---|---|
| Max-Sigmoid | (1-\sigma(z_{(1)})) | D 组原生置信度的补数。最大 Sigmoid 越低，越倾向于错分。 |
| Sigmoid Margin | (-[\sigma(z_{(1)})-\sigma(z_{(2)})]) | 前两类 Sigmoid 激活差越小，决策越模糊。 |
| Raw-logit Margin | (-(z_{(1)}-z_{(2)})) | 直接使用前两名 logit 间隔，避免 Sigmoid 饱和压缩。 |
| OVR Self-BCE | (K^{-1}[\operatorname{softplus}(-z_{\hat y})+\sum_{j\ne\hat y}\operatorname{softplus}(z_j)]) | 把模型自身预测当作伪标签，计算与 D 组训练目标完全同形的 one-vs-rest BCE。预测类不够正或其他类别仍被激活都会增大分数。 |
| DOCTOR | ((1-\sum_kp_k^2)/(\sum_kp_k^2)) | DOCTOR-α 的黑盒分数。为遵循原方法及便于和 C 组比较，这里对 D logits 重新做 Softmax，而不是把互不归一的 Sigmoid 当作类别分布。 |
| Energy | (-\log\sum_ke^{z_k}) | 温度固定为 (T=1)。分数越高表示整体证据越弱。 |
| L2Norm-MaxLogit | (-\max_k[z_k/\lVert z\rVert_2]) | 先逐样本进行固定的 L2 logit 归一化，再取最大 logit；用于削弱样本间 logit 范数漂移。无需训练或验证集调参。 |

### 数值实现要点

1. OVR Self-BCE 使用 `softplus` 的稳定形式，不显式计算 `log(sigmoid)`；除以类别数 (K) 与 D 组 `bce_reduction=mean` 一致。由于所有样本的 (K=4271) 相同，除以 (K) 不改变排序指标。
2. DOCTOR 通过 `log_softmax` 计算概率，并对分母作浮点下界保护。
3. L2Norm-MaxLogit 对零范数作下界保护；全零 logits 的归一化最大 logit 定义为0。
4. 推理固定为 FP32，不使用 autocast，减少 Sigmoid 饱和和低精度并列对排序指标的影响。
5. Max-Sigmoid 与未归一化 MaxLogit 是严格单调对应，理论上产生相同排序；因此本实验不重复列普通 MaxLogit，而使用非等价的 L2Norm-MaxLogit。

## 3. 评价指标与结果格式

| 指标 | 方向 | 定义 |
|---|:---:|---|
| AUROC | ↑ | 错误为正类的 ROC 曲线面积。 |
| Error AUPRC | ↑ | 错误为正类的 Precision–Recall 曲线面积。 |
| FPR@95TPR | ↓ | 错误检出率达到95%时，正确预测被误报为错误的比例；使用线性插值。 |
| AURC | ↓ | 按从安全到危险逐步接受样本的 Risk–Coverage 曲线面积；同分采用随机排列的期望风险。 |
| OSCR | ↑ | 闭集错分适配版：CCR 对错误预测错误接受率的曲线面积。它不是需要额外 OOD 数据的标准开放集 OSCR。 |

最终汇总严格按以下规则输出：

```text
reported_value = raw_value * 100
```

随后采用十进制 `ROUND_HALF_UP` 四舍五入并固定保留两位小数。例如：

```text
0.914884 -> 91.49
```

`metrics.csv`、`metrics.md`、两份曲线 CSV 及曲线坐标均使用百分制。`sample_scores.csv` 保存方法原始分数，不乘100，因为 Energy、OVR Self-BCE 和 margin 具有不同量纲；任意缩放这些原始分数没有科学意义，也不影响排序指标。`run_info.json` 同时记录原始准确率和百分制准确率，并明确记录缩放、舍入规则。

闭集适配 OSCR 满足：

```text
OSCR = D组准确率 × AUROC
```

因此固定 D 分类器时，它与 AUROC 给出相同的方法排序，不应被解释为独立证据。

## 4. 服务器运行方法

在服务器仓库根目录 `/mnt/hdd8t/Mingle/xyyy` 运行：

```bash
cd /mnt/hdd8t/Mingle/xyyy

python -m baselines.d_group_training_free_baselines \
  --config Classifier/outputs/ep20/l2_s42_retry1/config.json \
  --checkpoint Classifier/outputs/ep20/l2_s42_retry1/best.pt \
  --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/official_val \
  --output-dir /mnt/hdd8t/Mingle/xyyy/baselines/outputs/d_group/official_val \
  --device cuda
```

如果 D 组真实成功 run 的目录不是 `l2_s42_retry1`，必须将 `--config` 和 `--checkpoint` 改为真实目录，不能复制或猜测 checkpoint。

如果配置中的骨干或类别映射路径因迁移而失效，可显式覆盖：

```bash
python -m baselines.d_group_training_free_baselines \
  --config Classifier/outputs/ep20/l2_s42_retry1/config.json \
  --checkpoint Classifier/outputs/ep20/l2_s42_retry1/best.pt \
  --pretrained /mnt/hdd8t/Mingle/xyyy/models/misclassification-aware/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth \
  --class-map /mnt/hdd8t/Mingle/xyyy/MisD/data/class_to_idx.json \
  --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/official_val \
  --output-dir /mnt/hdd8t/Mingle/xyyy/baselines/outputs/d_group/official_val \
  --workers 8 \
  --device cuda
```

脚本会强制检查：

1. checkpoint 必须命名为 `best.pt`；
2. D 组必须是 LoRA+Head、L2/one-vs-rest BCE、seed=42、20 epochs、batch 32、rank 8、alpha 16；
3. checkpoint 内配置与外部 `config.json` 必须一致；
4. 类别映射和 PlantCLEF 骨干 SHA256 必须与训练 checkpoint 一致；
5. 输出目录必须不存在，避免覆盖正式结果；
6. 测试集必须同时包含正确和错误预测。

若 checkpoint 同目录包含 `history.jsonl`，脚本还会核验 `best.pt` 是否为最早达到最高 `classifier_val` Top-1 的 epoch。只加载自己训练并信任的 checkpoint。

## 5. 固定输出结构

```text
/mnt/hdd8t/Mingle/xyyy/baselines/outputs/d_group/official_val/
├── metrics.csv
├── metrics.md
├── sample_scores.csv
├── risk_coverage.csv
├── risk_coverage.png
├── oscr_curve.csv
├── oscr_curve.png
└── run_info.json
```

该结构与 C 组一致：

- `metrics.csv`：七种方法的百分制主结果，固定保留两位小数；
- `metrics.md`：便于直接查看和粘贴的百分制表格；
- `sample_scores.csv`：逐样本标签、预测、错误标记和七个原始错分分数；
- `risk_coverage.csv/png`：百分制 Risk–Coverage 数据与图；
- `oscr_curve.csv/png`：百分制闭集 OSCR 适配曲线；
- `run_info.json`：样本数、错误率、路径、checkpoint/backbone SHA256、公式、软件版本和审计信息。

## 6. 科研使用约束

1. `classifier_val` 只用于选择 D 组 `best.pt`；七种方法不得根据 `official_val` 结果增删或改公式。
2. 所有方法均无拟合参数。Energy 固定 (T=1)，L2Norm 固定 (p=2)。
3. 七种分数来自同一个模型、相同样本顺序和同一次 FP32 推理。
4. AUROC、AUPRC、FPR@95TPR 都以分类错误为正类。
5. Error AUPRC 必须与错误率一起报告；跨 C/D 比较时错误率可能不同。
6. AURC 同时受分类准确率和置信排序影响。C/D 准确率不同，不能仅凭 AURC 判断评分函数本身更好。
7. 当前只有 seed=42，结果属于受控描述性比较，不能宣称训练过程具有跨 seed 统计显著性。

## 7. 本地测试

从仓库根目录运行：

```bash
python -m unittest baselines.test_d_group_training_free_baselines -v
```

测试覆盖七种公式和方向、OVR Self-BCE 等价式、L2 零范数保护、Max-Sigmoid/MaxLogit 排序等价性、完美错误排序、AURC 同分处理、闭集 OSCR 恒等式、D 协议检查，以及百分制 `ROUND_HALF_UP` 两位小数输出。

## 8. 方法依据

- Hendrycks, D. and Gimpel, K. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. ICLR 2017.
- Granese, F. et al. *DOCTOR: A Simple Method for Detecting Misclassification Errors*. NeurIPS 2021.
- Liu, W. et al. *Energy-based Out-of-distribution Detection*. NeurIPS 2020.
- Padhy, S. et al. *Revisiting One-vs-All Classifiers for Predictive Uncertainty and Out-of-Distribution Detection in Neural Networks*. ICML Workshop 2020.
- Cattelan, L. F. P. and Silva, D. *How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks*. UAI 2024.

其中 OVR Self-BCE 是针对本项目 D 组训练目标构造的无训练诊断分数，应标注为项目中的 D-specific score，而不是冒充已有论文提出的标准命名基线。
