# Stage-1: Classification Model Training (LoRA + Sigmoid + Multi-Loss)

## 概述

本目录实现第一阶段分类模型训练实验。基于 PlantCLEF2022 MAE 预训练的 ViT-L/16，使用 LoRA 参数高效微调，输出 4271 个独立 Sigmoid 概率，比较三种 Loss 配置对分类性能的影响。

## 实验矩阵

| 实验 | Loss 配置 | λ1 (L1) | λ2 (L2) | λ3 (L3) | 采样方式 | Batch Size |
|------|-----------|---------|---------|---------|----------|------------|
| Exp-L1 | L1 (BCE) | 1 | 0 | 0 | 随机 | 128 |
| Exp-L1L3 | L1 + L3 (BCE + SupCon) | 1 | 0 | 1 | PK-Sampler | 128 (P=32, K=4) |
| Exp-L2L3 | L2 + L3 (Margin + SupCon) | 0 | 1 | 1 | PK-Sampler | 128 (P=32, K=4) |

## 损失函数

### L1: 独立 Sigmoid BCE
```
L1 = (1/C) Σ_i BCEWithLogits(z_i, y_i_onehot)
```
每个类别节点独立计算 BCE，不加 Softmax 归一化约束。

### L2: Indicator Margin Ranking
```
L2 = max(0, m - (z_y - max_{j≠y} z_j))
```
当正确类 logit 未领先最强竞争类至少 margin m=0.5 时产生惩罚。

### L3: Supervised Contrastive Loss
```
L3 = Σ_i (-1/|P(i)|) Σ_{p∈P(i)} log[exp(f_i·f_p/τ) / Σ_a exp(f_i·f_a/τ)]
```
在 L2 归一化的 1024 维特征空间上计算，温度 τ=0.07。配合 PK-Sampler 确保每个 batch 内同类样本存在。

## 模型结构

```
PlantCLEF MAE ViT-L/16 (预训练)
    ↓
LoRA (rank=16, alpha=32, dropout=0.1, targets=[qkv, proj])
    ↓
Global Pool → fc_norm → 1024 维特征
    ↓
BatchNorm1d(affine=False) → Linear(1024, 4271)
    ↓
4271 维 logits → Sigmoid → 4271 维概率
    ↓
prediction = argmax(p),  confidence = max(p)
```

## PEFT 参数

| 参数 | 值 |
|------|-----|
| Method | LoRA |
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.1 |
| Target Modules | qkv, proj (每个 attention block) |
| Trainable Params | ~LoRA params + 分类头 |
| Total Params | ~304M (ViT-L/16) |

## 训练超参数

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-3 |
| Epochs | 50 |
| Warmup Epochs | 5 |
| LR Schedule | Cosine decay |
| Batch Size | 128 |
| AMP | 开启 (fp16) |
| Gradient Clipping | 1.0 (global norm) |
| Random Seed | 0 |

## 代码结构

```
Classifier/
├── models.py            # ViT + LoRA + Sigmoid 分类头
├── losses.py            # L1 (BCE), L2 (Margin), L3 (SupCon), CombinedLoss
├── data.py              # 数据加载 + PK-Sampler
├── train.py             # 主训练脚本
├── export_outputs.py    # 导出完整 4271 维输出
├── run_exp_l1.sh        # Exp-L1 运行脚本
├── run_exp_l1_l3.sh     # Exp-L1L3 运行脚本
├── run_exp_l2_l3.sh     # Exp-L2L3 运行脚本
├── run_all.sh           # 一键运行三组实验
├── requirements.txt     # 依赖
└── output/              # 实验输出（gitignore）
    ├── exp_l1/
    │   ├── config.json
    │   ├── metrics.csv / metrics.jsonl
    │   ├── tensorboard/
    │   ├── checkpoint_best.pth
    │   ├── checkpoint_last.pth
    │   ├── predictions_val.npz    # 完整 4271 维输出
    │   └── predictions_val.csv    # 每样本摘要
    ├── exp_l1_l3/
    └── exp_l2_l3/
```

## 运行方式

### 前提条件
- 服务器上已有 iNaturalist 2021 Plant 数据集（ImageFolder 格式）
- 服务器上已有 PlantCLEF2022 MAE 预训练权重
- 已安装 PyTorch + timm

### 单独运行

```bash
# Exp-L1 (BCE)
bash Classifier/run_exp_l1.sh

# Exp-L1L3 (BCE + SupCon)
bash Classifier/run_exp_l1_l3.sh

# Exp-L2L3 (Margin + SupCon)
bash Classifier/run_exp_l2_l3.sh
```

### 一键运行全部

```bash
bash Classifier/run_all.sh
```

### 自定义路径

```bash
export PROJECT_ROOT=/your/path
export CUDA_VISIBLE_DEVICES=0,1
bash Classifier/run_all.sh
```

## 输出说明

### predictions_val.npz
| 字段 | 形状 | 说明 |
|------|------|------|
| ground_truth | [N] | 真实类别索引 |
| prediction | [N] | 预测类别 (argmax) |
| p_max | [N] | 最大 Sigmoid 概率 |
| probs | [N, 4271] | 完整 4271 维 Sigmoid 概率 |
| sample_paths | [N] | 样本文件路径 |

### predictions_val.csv
```
sample_path, ground_truth, prediction, p_max, correct
```

## 评价指标

- **Top-1 Accuracy** (主指标): 预测类别 == 真实类别
- **Top-5 Accuracy**: 真实类别在 Top-5 预测中
- **Train Loss / 各 Loss 分量**: 观察收敛情况
- **置信度分布**: 为第二阶段错分检测准备

## 与第二阶段的衔接

第一阶段完成后，每个实验目录下的 `predictions_val.npz` 包含第二阶段所需的全部数据：
- 完整 4271 维 Sigmoid 概率（用于 Max Probability + Threshold 基线）
- Ground Truth + Prediction（定义错分 Ground Truth）
- p_max（最大置信度）

根据 To_do_list 第30节，模型选择不仅看 Accuracy，还需综合考虑置信度分布、高置信度错误和低置信度正确样本的比例。
