# C 组模型：无需额外训练的错分检测基线

## 1. 实验目的与范围

本实验固定使用 A/B/C/D 受控对比中的 **C 组最佳模型**：LoRA+Head、Cross-Entropy、seed=42，并使用 `classifier_val` Top-1 选择的 `best.pt`。脚本不会训练或拟合任何新的参数；七种置信度/不确定性分数均由同一次前向产生的 logits 直接计算。

错误预测定义为正类：

```text
error_label = 1[prediction != ground_truth]
```

所有导出的 `*_error_score` 都统一为：数值越大，样本越可能被错分。

## 2. 七种基线定义

记 logits 为 \(z\in\mathbb{R}^K\)，\(p=\operatorname{softmax}(z)\)。\(z_{(1)},z_{(2)}\) 与 \(p_{(1)},p_{(2)}\) 分别表示最大值和次大值。

| 方法 | 本实现的错分分数 | 含义 |
|---|---|---|
| MSP | \(1-p_{(1)}\) | 最大 softmax 概率越小，预测越不可信。它是最经典的错分检测基线。 |
| Probability Margin | \(-(p_{(1)}-p_{(2)})\) | 第一、第二候选类别的概率差越小，决策越模糊。负号只用于统一分数方向。 |
| Entropy | \(-\sum_k p_k\log p_k\) | 整个预测分布越分散，预测不确定性越高。 |
| DOCTOR | \((1-\sum_kp_k^2)/(\sum_kp_k^2)\) | DOCTOR-α 的完全黑盒形式，利用全部 softmax 概率衡量碰撞概率/二阶 Rényi 不确定性。 |
| Raw-logit Margin | \(-(z_{(1)}-z_{(2)})\) | 直接衡量前两名原始 logits 的决策间隔，避免 softmax 分母与尺度压缩带来的影响。 |
| MaxLogit | \(-z_{(1)}\) | 最大原始 logit 越小，越倾向于视为错误。 |
| Energy | \(-\log\sum_k e^{z_k}\) | 温度固定为 \(T=1\) 的标准 energy；能量越高（越不负），越倾向于错误。 |

Probability Margin 与 Raw-logit Margin 不是同一个量，必须分别报告。DOCTOR 使用原论文的 \(D_\alpha\) 排序分数；它与 \(1-\sum p_k^2\) 排序等价，但这里保留论文中的 odds 形式。

## 3. 评价指标

| 指标 | 方向 | 定义与注意事项 |
|---|:---:|---|
| AUROC | ↑ | 错误为正类，评价错误样本排在正确样本之前的概率；不依赖固定阈值。 |
| Error AUPRC | ↑ | 错误为正类的 Precision–Recall 曲线面积；受测试集错误率影响，必须同时报告 `run_info.json` 中的 `error_prevalence`。 |
| FPR@95TPR | ↓ | 错误检出率 TPR=95% 时，正确样本被误报为错误的比例；采用 ROC 曲线线性插值。 |
| AURC | ↓ | 按从安全到危险的顺序逐步接受样本，计算 Risk–Coverage 曲线面积；同分样本采用随机排列的期望风险，避免文件顺序造成偏差。 |
| OSCR | ↑ | 本实验的**闭集错分适配版**：横轴是错误预测被错误接受的比例，纵轴是“正确且被接受”的样本占全部测试样本的比例，报告曲线面积。 |

### OSCR 的必要限定

标准 Open-Set Classification Rate 需要一组已知类测试样本和一组独立 OOD/unknown 样本。本实验目前只有闭集标签，因此不能声称计算了标准开放集 OSCR。脚本把“模型已经产生的错误预测”当作需要拒绝的负样本，输出列仍命名为 `oscr`，但 `run_info.json` 和曲线坐标均明确标记为 closed-set adaptation。

在这个定义下：

```text
OSCR = C 组测试准确率 × 错分检测 AUROC
```

所以在固定 C 组分类器、只比较七种分数时，OSCR 与 AUROC 给出相同的方法排序；它不是独立的错分排序证据。如果后续加入真正 OOD 数据，应另写标准 open-set OSCR 实验，不能与本表混合。

## 4. 服务器运行方法

必须从仓库根目录运行。下面以最终测试集 `official_val` 为例：

```bash
python -m baselines.c_group_training_free_baselines \
  --config Classifier/outputs/abc_seed42/C/config.json \
  --checkpoint Classifier/outputs/abc_seed42/C/best.pt \
  --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/official_val \
  --output-dir baselines/outputs/c_group/official_val \
  --device cuda
```

如果代码仓库或权重迁移后路径发生变化，可显式覆盖路径：

```bash
python -m baselines.c_group_training_free_baselines \
  --config Classifier/outputs/abc_seed42/C/config.json \
  --checkpoint Classifier/outputs/abc_seed42/C/best.pt \
  --pretrained /mnt/hdd8t/Mingle/xyyy/models/misclassification-aware/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth \
  --class-map /mnt/hdd8t/Mingle/xyyy/MisD/data/class_to_idx.json \
  --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/official_val \
  --output-dir baselines/outputs/c_group/official_val \
  --workers 8 \
  --device cuda
```

脚本会进行以下强制检查：

1. checkpoint 文件名必须为 `best.pt`；
2. checkpoint 和配置必须是 C 组、LoRA+Head、CE、seed=42；
3. 类别映射必须与训练 checkpoint 完全一致；
4. PlantCLEF 骨干的 SHA256 必须与训练时一致；
5. 测试集必须包含全部 4271 个类别目录并使用相同映射；
6. 输出目录必须不存在，防止无意覆盖旧结果；
7. 测试集中必须同时存在正确和错误预测。

如果 checkpoint 同目录存在 `history.jsonl`，脚本还会验证 `best.pt` 是否确实对应最早达到最高 `classifier_val` Top-1 的 epoch；如果历史文件未随模型复制，实验仍可运行，但 `run_info.json` 会把该项记录为未验证。

只加载自己训练并信任的 checkpoint；PyTorch 的 `weights_only=False` 会反序列化 checkpoint 中保存的 RNG 状态。

## 5. 固定输出结构

```text
baselines/outputs/c_group/official_val/
├── metrics.csv
├── metrics.md
├── sample_scores.csv
├── risk_coverage.csv
├── risk_coverage.png
├── oscr_curve.csv
├── oscr_curve.png
└── run_info.json
```

- `metrics.csv`：机器可读的主结果表，列顺序固定为 method、AUROC、Error AUPRC、FPR@95TPR、AURC、OSCR。
- `metrics.md`：与 CSV 数值一致、保留六位小数的论文检查表。
- `sample_scores.csv`：逐样本标签、预测、错误标记和七个原始错分分数，用于复核与 bootstrap。
- `risk_coverage.csv/png`：七种方法的 Risk–Coverage 数据与图。
- `oscr_curve.csv/png`：闭集 OSCR 适配曲线；坐标名称不会伪装成标准 OOD OSCR。
- `run_info.json`：样本数、错误率、checkpoint/backbone SHA256、epoch、路径、Git revision、公式与评价定义。

## 6. 科研使用约束

1. `classifier_val` 只用于选择 C 组的 `best.pt`；不得根据 `official_val` 结果重新选择模型、分数、温度或阈值。
2. 七种方法都没有拟合超参数。Energy 固定 \(T=1\)，不在最终测试集调温度。
3. 七种分数来自同一模型、同一批样本、同一次确定性中心裁剪推理，预测均为 `argmax(logits)`。
4. AUROC/AUPRC/FPR@95TPR 将错误作为正类；不要与“正确为正类”的论文结果直接比较。
5. Error AUPRC 必须和错误率一起报告。跨数据集或跨分类器比较 AUPRC 时尤其需要注意基准错误比例变化。
6. AURC 同时受分类准确率和置信排序影响；本表固定同一个 C 模型，因此七种方法之间可以直接比较。
7. 当前实验只有 seed=42。七种分数作用于同一组预测，但不能据此宣称训练过程具有跨 seed 统计显著性。

## 7. 运行测试

```bash
python -m unittest baselines.test_c_group_training_free_baselines -v
```

测试覆盖分数公式与方向、完美错误排序、AURC 同分处理、以及闭集 OSCR 与 `accuracy × AUROC` 的一致性。

## 8. 方法依据

- Hendrycks, D. and Gimpel, K. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. ICLR 2017.
- Granese, F. et al. *DOCTOR: A Simple Method for Detecting Misclassification Errors*. NeurIPS 2021.
- Liu, W. et al. *Energy-based Out-of-distribution Detection*. NeurIPS 2020.
- Liang, H., Peng, L. and Sun, J. *Selective Classification Under Distribution Shifts*. TMLR 2024.
- Dhamija, A. R. et al. *Reducing Network Agnostophobia*. NeurIPS 2018.（标准开放集 OSCR 来源；本实验仅使用明确标注的闭集适配版。）
