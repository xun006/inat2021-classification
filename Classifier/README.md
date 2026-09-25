# 第一阶段：PlantCLEF ViT + LoRA + 四组 Loss 对照

目标：在 iNaturalist2021 Plant 的 4271 类上训练分类器，比较分类能力与错分排序能力，为阶段二输出完整数据。实现依据需求文档第 37–48 节。这里是待运行的实验方案与代码，不包含真实实验结果。

## 实验流程与固定设置

1. **确认数据划分**：复用 `MisD/tool/build_dataset_splits.py` 的五路隔离方案。源训练集每类抽取 classifier_val=10、detector_train=30、detector_calibration=5，余下用于 classifier_train；官方 val 留作最终测试。不要把 detector 数据并回分类训练，也不要用官方 val 调参。已有划分直接使用，不重新随机划分。划分脚本可能允许恰好 45 张的类别产生空训练目录，本代码会拒绝这种数据，需先修正划分。
2. **预检查**：检查类映射、训练/验证样本交叉和类别频率。`audit` 检查相同相对文件名、硬链接与软链接复用，不做图像内容哈希；重命名复制图、同一 observation 多图需要结合原始元数据另做泄漏审计。数据实际规模、长尾程度由统计决定，不能预先认定一定长尾。
3. **小规模验收**：先运行单元测试；随后在真实服务器复制配置，设 epochs=2、lr_warmup=0、l3_warmup=0、l3_ramp=1，单独输出 smoke 目录，确认权重加载、显存与导出。它不计入正式结果。
4. **先跑基线**：seed=42 的 L1、L2。检查 loss 曲线、正负 BCE 项、每项分类头梯度范数。BCE 固定为 `mean over all batch/class elements`，因此负项可能占主导，不默认加正类权重。
5. **再跑组合**：相同配置与 seed 跑 L1+L3、L2+L3。只改变 loss。L3 第 1–3 epoch 关闭，第 4–6 epoch 权重 1/3、2/3、1，后续为 1。若某项明显主导梯度，先记录，再以新实验 ID 在验证集比较 lambda3={0.1,0.5,1}。
6. **重复验证**：四组均补 seed=43、44，报告每次结果与均值/样本标准差。四组总计 12 次训练。初始化、样本顺序和增强在同 seed 下相同；修改 batch size 要对四组统一修改，尤其 L3 依赖物理 batch 中的正确/错误配对。
7. **选择模型**：每次训练 best.pt 以 classifier_val Top-1 最高选 epoch（同分取最早）。最终跨配置比较 Top-1、Macro-F1、各频率组、AUROC、AUPR-Error、AURC 及高置信错误。建议预先把可接受 Top-1 降幅设为 0.5 个百分点，作为待验证的研究约束；在满足约束者中检查排序指标与多 seed 稳定性，记录最终选择理由。不可只看某一检测指标或测试集挑模型。
8. **固定模型后**：显式导出 detector_train、detector_calibration 和 official_val。阶段一选择期间仅导出 classifier_val。官方 val 最终评估一次；所有阈值与超参数用验证集决定。

默认起点：ViT-L/16、224px、平均池化；每层 fused attention QKV 加 LoRA（包含 Q/K/V 三段），rank=8、alpha=16、dropout=0；骨干冻结、线性分类头重新初始化。AdamW，LoRA LR=1e-4、head LR=1e-3、weight_decay=0.05、30 epoch、batch=32、LR 预热 3 epoch 后 cosine、梯度裁剪 1、CUDA FP16 训练/FP32 验证。增强固定 RandomResizedCrop(0.5–1.0)+水平翻转；不使用 Mixup/CutMix，保持单标签与 L3 正误分组含义。

这是**单进程单 GPU**实现，一张卡一个实验；多张卡可以分别启动独立实验。不要用 torchrun：已显式拒绝 WORLD_SIZE>1。没有用梯度累积模拟更大的 L3 batch。ViT-L 的实际显存须在服务器测量，OOM 时统一减小四组 batch。

## 服务器安装与配置

从仓库根目录执行，Python 3.10/3.11 推荐。建议创建独立环境，避免影响旧版 PlantCLEF 实验。

```bash
# 先按服务器驱动安装配套 CUDA torch/torchvision，再安装其余依赖
python -m pip install -r Classifier/requirements.txt
cp Classifier/config.json Classifier/config.local.json
# 默认路径已由用户确认；仅在服务器目录变化时编辑 config.local.json
python -m unittest Classifier.test_stage1 -v
python -m Classifier.run audit --config Classifier/config.local.json --output Classifier/outputs/audit
```

首次正式实验建议给 audit 加 `--verify-images`，逐图解码排查损坏并保存宽高分位数（全量读取较慢）。训练和导出遇到损坏图片均报错，不静默跳过样本。

路径支持仓库外的数据与模型；运行位置固定为仓库根目录。用户已确认数据根目录为 `/mnt/hdd8t/Mingle/xyyy/MisD/data`，预训练权重为 `/mnt/hdd8t/Mingle/xyyy/models/misclassification-aware/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth`，与 `config.json` 默认值一致。数据根目录下使用 `classifier_train`、`classifier_val` 和 `class_to_idx.json`；运行 audit 检查这些子目录和类别映射。如果实际布局为 `classifier/train`、`classifier/val`，只改配置中的对应路径。每个划分必须具有一致的全部类目录，字典映射必须与 ImageFolder 字典序一致。文件名里的下划线不需要添加反斜杠。

权重支持原始 state_dict 或 `model`/`state_dict` 包装，移除 `module.` 前缀；明确丢弃 MAE decoder/mask_token 与预训练 head。平均池化时将 encoder norm 转为 fc_norm。除 head 外任何缺失、多余键或形状不匹配直接报错，避免误用 4271 类旧 linear-probe checkpoint 代替 PlantCLEF 初始化。当前不插值位置编码，真实实验使用与权重匹配的 224px；不能只改 image_size 就声称可用。

## 训练、恢复与导出

```bash
python -m Classifier.run train --config Classifier/config.local.json \
  --loss l1 --seed 42 --output Classifier/outputs/l1_s42

# 同一配置恢复，必须使用该目录 last.pt；从完整 epoch 边界恢复
python -m Classifier.run train --config Classifier/config.local.json \
  --loss l1 --seed 42 --output Classifier/outputs/l1_s42 \
  --checkpoint Classifier/outputs/l1_s42/last.pt

# 导出时使用训练时保存的配置，避免 loss/seed 等不一致
python -m Classifier.run export --config Classifier/outputs/l1_s42/config.json \
  --checkpoint Classifier/outputs/l1_s42/best.pt \
  --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/classifier_val \
  --output Classifier/outputs/l1_s42/val_export
```

一轮基线检查完毕后可执行四组正式矩阵；已存在的目录会拒绝覆盖，可手动删去已经完成的循环项：

```bash
set -euo pipefail
for seed in 42 43 44; do
  for loss in l1 l2 l1_l3 l2_l3; do
    run="Classifier/outputs/${loss}_s${seed}"
    python -m Classifier.run train --config Classifier/config.local.json \
      --loss "$loss" --seed "$seed" --output "$run"
    python -m Classifier.run export --config "$run/config.json" \
      --checkpoint "$run/best.pt" --split-dir /mnt/hdd8t/Mingle/xyyy/MisD/data/classifier_val \
      --output "$run/val_export"
  done
done
python -m Classifier.summarize Classifier/outputs --output Classifier/outputs/comparison
```

多 GPU 运行可用 `CUDA_VISIBLE_DEVICES=1 python ...` 给不同实验分配不同卡，输出目录必须不同。训练配置不可在恢复时改变；数据/权重可迁移路径，但类映射、划分指纹、预训练 SHA256 必须一致。恢复含优化器、GradScaler、Python/NumPy/Torch/CUDA RNG 状态；不承诺跨硬件/软件版本的逐位一致性。仅加载自己生成、可信的 adapter checkpoint（其中含 Python RNG 对象）。

## 产物及指标约定

- `best.pt` / `last.pt`：LoRA+head、优化器、RNG、配置、类映射、数据统计、预训练权重 SHA256。**不是自包含完整骨干**，重建时仍需同一份 PlantCLEF 权重。
- `config.json` / `model_report.json` / `data_audit.json`：训练设置、结构、参数量/比例、实际 target modules、版本、Git revision、样本量、每类样本数、划分指纹。
- `history.jsonl`：逐 epoch 的训练与验证分项 loss、分类与检测指标；`diagnostics.jsonl`：每 epoch 前 5 batch 的 L1/L2/L3 分类头梯度范数、BCE 正负项、真实/最强错误 logit、有效配对数和饱和比例。验证 L3 仅在验证 batch 内计算，依赖 batch 划分，只作诊断。
- `logits.npy` / `sigmoid.npy`：`N×4271` float32，磁盘 memmap 分批写入；每个样本两者合计 34168 bytes，10 万样本约 3.18 GiB。不会在 RAM 中堆积完整矩阵。
- `samples.csv`：行索引、相对路径 sample_id、标签、预测、正确标记、最大 Sigmoid、top1/top2 logits 及 margin、真实类 logit、最强错误类 logit 及 margin、频率组。和两个矩阵逐行对应。
- `metrics.json`、`per_class.csv`、`risk_coverage.npz`、`provenance.json`：完整评估、逐类 accuracy、风险覆盖曲线、导出模型身份与映射。`complete.json` 仅成功结束后出现；未完成导出不可当正式结果使用，改用新目录重新导出。
- `comparison/runs.csv`、`summary.json`：按 loss 汇总验证导出。请只汇总同一固定设置的实验；不要把不同 batch、lambda 或超参数的结果混在同一目录。

预测用 argmax(logits)，数学上等价于 Sigmoid argmax，且避免浮点 Sigmoid 饱和造成伪并列。最大 Sigmoid 是置信分数，未经校准不等于预测正确概率。检测正类=分类错误，error_score=1-max_sigmoid。AUPR-Error 使用 sklearn average precision；AURC 使用离散覆盖率风险均值，置信度并列时计算组内随机顺序的期望风险。全对/全错时 AUROC 为 null，全对时 AUPR 为 null。

Head/Medium/Tail 按**分类训练集**计数：>100 / 20–100 / <20，空组返回 null；Macro-F1 固定包含全部 C 类。tau 默认 0.5 仅作描述性统计，高置信错分和低置信正确数同时保存，不用它选择测试集阈值。分数直方图为 [0,1] 的 20 个等宽区间；逐样本 margin 可从 CSV 再分析。

## 当前验证与边界

本机没有真实数据集与 PlantCLEF 权重，因此无法报告准确率、真实显存占用或正式训练耗时。合成测试仅验证实现链路，不代表真实模型性能。服务器首次运行须保留权重加载报告、audit、测试结果和 smoke 输出；确认这些内容后再投入 12 组完整训练。

2026-09-25 本地 7 项测试全部通过：损失公式/梯度、四组优化器更新、L3 warmup、并列分数指标、模拟 MAE 权重加载、LoRA 参数冻结、平均池化前向一致性、两 epoch 训练与断点恢复参数逐位一致、导出矩阵检查及划分交叉拒绝。环境为现有 Python 3.8.20 / torch 2.4.1 CPU + 工作区隔离安装的 timm 1.0.22、sklearn 1.3.2；服务器推荐环境仍为 Python 3.10/3.11。语法编译、CLI 帮助、Git diff 空白检查通过。

实现参考：[PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)、[timm 模型接口](https://huggingface.co/docs/timm/reference/models)。BCE 在 logits 上计算，reduction 明确固定；不直接对 argmax 指示函数求导。

## GitHub 与交接

`Classifier/.gitignore` 已排除模型、outputs、预测矩阵、本机配置与测试依赖。代码未自动 commit/push；准备提交时检查 `git status`，只加入 Classifier 源码及文档。外部 `Hand_off.md` 不在当前仓库，需另行保管。每次迭代向其**末尾追加**目的、设计决策、代码变化、实际运行/未运行项、结果、问题、下一步，不改写历史。
