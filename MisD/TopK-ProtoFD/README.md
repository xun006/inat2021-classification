# Top-K Prototype Failure Detector

冻结已训练的 ViT-L/16，默认使用分类器训练集生成的类别均值特征作为 L2 归一化类别原型。图像分类头输入特征作为 query，概率最高的 K 个类别原型作为 key/value；检测头仅输入 `[图像特征, 预测类原型, attention context]`，输出错分概率。

## 数据布局

数据按 torchvision `ImageFolder` 组织，三个 split 必须有完全相同的类别目录与类别索引：

```text
DATA_ROOT/
├── detector_train/<class_name>/*.jpg
├── detector_calibration/<class_name>/*.jpg
└── official_val/<class_name>/*.jpg
```

## 1. 构建分类器训练集均值原型

只使用 `/mnt/hdd8t/Mingle/xyyy/MisD/data/classifier/train`，不使用 classifier val：

```bash
python build_mean_prototypes.py \
  --batch-size 128 \
  --device cuda
```

默认输出为 `prototypes/classifier_train_mean.pth`，包含 `[4271, 1024]` 原型、每类样本数、类别名称和来源信息。每张图像特征先 L2 归一化，按真实类别求均值后再次 L2 归一化。

## 2. 训练 Failure Detector

```bash
python train.py \
  --data-root /path/to/data \
  --batch-size 128 \
  --device cuda
```

当前默认实验使用均值原型和 bottleneck detector：K=5、attention dim=256，图像特征由 `1024→128`，预测类均值原型由 `1024→128`，attention context 保持 256 维。检测头只输入这三个部分，不加入 Top-K 概率、余弦相似度或 attention 权重等额外统计量。训练参数为 dropout=0.3、AdamW lr=1e-4、weight decay=1e-3、gradient clipping=1.0；calibration AUROC 停滞 2 轮后学习率减半，连续 6 轮无实质提升后 early stop。

训练结果默认写入 `outputs/mean_proto_bottleneck_k5/`。`detector_best.pth` 会直接包含均值原型及其来源元数据，评估时不会回退到分类器权重。

若后续需要复现不降维的原始结构，可显式使用 legacy：

```bash
python train.py \
  --data-root /path/to/data \
  --architecture legacy \
  --dropout 0.1 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --output-dir outputs/mean_proto_legacy_k5
```

## 单独评估

```bash
python evaluate.py \
  --detector-checkpoint outputs/mean_proto_bottleneck_k5/detector_best.pth \
  --data-root /path/to/data \
  --split official_val \
  --device cuda
```

输出指标包括 AUROC、FPR95、AUPR-Error、AURC 和基础分类错误率。错误标签始终由冻结 ViT 的当前预测与真实标签在线生成。

## 重要约定

- checkpoint 已适配实际结构：global-pool ViT-L/16，1024 维特征，4271 类，`BatchNorm1d(affine=False) + Linear` 分类头。
- Top-K 类别仍由原分类 logits 决定；只有 attention 使用的 prototype key/value 以及预测类原型被替换为数据均值原型，原分类结果完全不变。
- 分类 logits 与训练原模型保持不变；ViT 参数以及分类头 BatchNorm 统计量均被冻结。
- batch 正样本权重严格使用 `log(1 + N_correct / max(N_error, 1))`；无错误样本时正项不存在。
- 数据预处理为 ImageNet mean/std、resize 256、center crop 224。若原分类器评估预处理不同，请同步修改 `topk_proto_fd/data.py`。
