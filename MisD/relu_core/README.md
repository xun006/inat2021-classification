# REL-U-Core：Classifier C 严格对比实验

本目录实现 ICLR 2024 *A Data-Driven Measure of Relative Uncertainty for
Misclassification Detection* 的 REL-U-Core。实现只读取仓库中的冻结 Classifier C、
数据和类别映射；所有新增代码均位于 `MisD/relu_core/`，运行结果默认写入
`MisD/output/relu_core_classifier_c_seed42/`。

## 1. 锁定的实验协议

- 基础模型：受控实验 C 组，LoRA+Head、Cross-Entropy、seed=42、`best.pt`。
- `detector_train`（128,130 张）：只用于估计 REL-U 的正确/错误二阶矩。
- `detector_calibration`（21,355 张）：只用于选择 lambda 和 temperature。
- `official_val`（42,710 张）：配置锁定后只进行一次最终评价。
- 错误定义始终为原始 Classifier C 的 `argmax(logits) != ground_truth`。
- epsilon 固定为 0；不进行输入扰动，不改变基础分类器预测。
- 主选择指标固定为 calibration Error-AUPRC；分数越大表示越可能错分。
- 默认网格在运行前写死：
  - lambda：`0, 0.1, ..., 1.0`
  - temperature：`0.5, 0.75, 1.0, 1.25, 1.5, 2.0`
- 决胜顺序固定为 Error-AUPRC、AUROC、temperature 距离 1、lambda 距离训练集正确率。

`fit_select.py` 无法读取 `official_val`。`evaluate.py` 必须显式提供
`--allow-official-val`，并验证锁定模型、教师 checkpoint、配置文件和类别映射的内容身份。
所有输出目录均禁止覆盖。

## 2. 数学实现

对温度缩放后的完整概率向量 `p`，分别计算：

```text
M_correct = mean(p.T @ p | original prediction is correct)
M_error   = mean(p.T @ p | original prediction is wrong)
D_raw     = lambda * M_error - (1-lambda) * M_correct
D         = FrobeniusNormalize(ReLU(SymmetricZeroDiagonal(D_raw)))
score(x)  = p(x) D p(x)^T
```

代码不构造原作者示例中的 `N x C x C` 外积张量，而使用数学完全等价的分块
`P.T @ P`。对于 4,271 类，这避免数 TB 的临时内存；它不是 Top-k 或其他近似。
测试打分使用 `((P @ D) * P).sum(1)`，不构造 batch×batch 矩阵。

## 3. 运行前测试

从仓库根目录执行：

```bash
python -m unittest discover -s MisD/relu_core -p 'test_*.py' -v
```

## 4. 开发阶段：导出、拟合、选择

服务器路径与当前工程一致时：

```bash
bash /mnt/hdd8t/Mingle/xyyy/MisD/relu_core/run_development.sh
```

如路径不同，可通过环境变量覆盖，但不要修改脚本中的协议：

```bash
ROOT=/mnt/hdd8t/Mingle/xyyy \
CLASSIFIER_RUN=/mnt/hdd8t/Mingle/xyyy/Classifier/outputs/abc_seed42/C \
OUT=/mnt/hdd8t/Mingle/xyyy/MisD/output/relu_core_classifier_c_seed42 \
bash /mnt/hdd8t/Mingle/xyyy/MisD/relu_core/run_development.sh
```

开发阶段输出：

```text
OUT/
├── cache/
│   ├── detector_train/
│   │   ├── logits.npy
│   │   ├── targets.npy
│   │   ├── predictions.npy
│   │   ├── errors.npy
│   │   ├── sample_keys.txt
│   │   └── manifest.json
│   └── detector_calibration/...
└── selection/
    ├── selected_model.pt
    ├── selected_config.json
    ├── search_results.csv
    └── calibration_predictions.csv
```

完整 float32 train logits 约占 2.2 GB。缓存保存完整 4,271 类 logits，softmax 始终
以 float32 计算。矩阵乘法关闭 CUDA TF32，以减少硬件近似造成的排序差异。

## 5. 最终阶段：一次性 official_val

先检查 `selection/selected_config.json`、锁定实验记录和输出路径。确认不再调整任何
网格、方向或选择规则后运行：

```bash
bash /mnt/hdd8t/Mingle/xyyy/MisD/relu_core/run_final.sh
```

最终输出：

```text
OUT/final/selected/
├── predictions.csv
└── metrics.json
```

`metrics.json` 报告 AUROC、Error-AUPRC、FPR@95 Error-TPR、tie-aware AURC、
E-AURC、错误率、样本数及全部 provenance。注意，这里的 FPR@95 把错误作为正类，
与 REL-U 原论文把正确分类作为 positive 的表格定义不同，不能直接比较数值。

## 6. 禁止事项

1. 不得根据 `official_val` 修改 lambda、temperature、分数方向或候选网格。
2. 不得把 calibration 合并回 train 后重新拟合最终矩阵，除非所有学习型检测器均采用
   同样规则并重新设计实验协议。
3. 不得将概率截断为 Top-k 后仍命名为 REL-U-Core。
4. 不得从扰动后的 logits 重新定义错误标签。
5. 不得覆盖已有输出目录；新的复现实验应使用新的 `OUT` 名称。

## 7. 方法来源

- Eduardo Dadalto et al., *A Data-Driven Measure of Relative Uncertainty for
  Misclassification Detection*, ICLR 2024.
- 论文：https://proceedings.iclr.cc/paper_files/paper/2024/file/5e0da5da69b71349ae0bd7ad716e4bc9-Paper-Conference.pdf
- 官方代码：https://github.com/edadaltocg/relative-uncertainty
