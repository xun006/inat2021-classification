
---

## 2026-09-25 — Stage1-Implementation-01：阶段一实验设计与代码实现

### 实验目的

依据 To_do_list.md，尤其第 37–48 节，在固定 PlantCLEF ViT、PEFT、数据和训练设置的条件下，实现 L1 / L2 / L1+L3 / L2+L3 公平对照，为阶段二保存完整输出。本次交付代码和流程，未执行真实数据上的完整训练。

### 设计决策与实验设置

- 数据：复用现有五路隔离划分。classifier_train 训练、classifier_val 选择模型；detector_train / detector_calibration 留作阶段二；official_val 留作最终评估。路径外置，默认路径来自前置代码，服务器运行前须核实。
- 模型：PlantCLEF MAE ViT-L/16，224px、平均池化，重新初始化 4271 类线性头。独立现代 timm 实现，与原 PlantCLEF 平均池化计算进行合成一致性检查。加载权重时丢弃 decoder/mask_token/head，norm 转 fc_norm，其余不匹配报错。
- PEFT：每层 attention fused QKV LoRA（Q/K/V 全段），rank=8、alpha=16、dropout=0；冻结原参数，训练 LoRA 和 head。
- 优化：AdamW，LoRA LR=1e-4、head LR=1e-3、weight_decay=0.05、batch=32、30 epochs、3 epochs LR warmup + cosine、梯度裁剪 1。单 GPU/单进程，CUDA FP16 训练，FP32 验证；不支持 DDP，不用梯度累积改变 L3 配对语义。
- L1：mean relu(1 - true_logit + max_wrong_logit)。L2：one-hot BCEWithLogits，mean over all elements。L3：batch 内全部正确–错误配对的置信排序 hinge，m_conf=0.1，无有效配对时可微零。lambda3=1，前 3 epochs 关闭，随后 3 epochs 线性增加。
- 公平对照：同 seed 的模型初始化、数据顺序、增强、训练参数一致。先 seed=42 两个基线检查量级，再跑组合，最终补 seeds=43、44（12 次正式训练）。超参仅用 classifier_val 选择。
- 模型选择：每个 run 按验证 Top-1 选 best epoch；跨 run 综合分类指标、检测排序指标、置信错误与多 seed 稳定性。建议事先记录可接受准确率降幅（起点 0.5 个百分点），不从测试结果反推标准。

### 代码变更

全部新增于 Experiment/Classifier，未修改现有实验模块：

- config.json：服务器路径与全部固定参数。
- losses.py：L1/L2/L3、warmup/ramp。
- model.py：PlantCLEF 兼容加载、LoRA 和新分类头。
- data.py：严格类映射、划分重叠检查、频率统计、可选全量解码与尺寸分布。
- run.py：audit/train/export 命令，训练诊断、best/last adapter checkpoint、优化器与 RNG 恢复、完整矩阵 memmap 导出。
- metrics.py：Top-1/5、Macro-F1、Head/Medium/Tail、AUROC、AP-Error、并列分数期望 AURC、置信度统计。
- summarize.py：验证结果逐 run CSV、按 loss 多 seed 均值/标准差。
- test_stage1.py：7 项 CPU 测试（包含训练、恢复和导出链路）。
- README.md：中文实验方案、安装配置、单次/矩阵训练、恢复、导出、指标与限制说明。
- requirements.txt、.gitignore：依赖和大文件隔离；HANDOFF_APPEND.md 保存本次交接追加文本副本，便于随 Git 提交。

### 实际运行与结果

本机没有真实 PlantCLEF 权重和 iNat 图像，未执行真实正式训练，没有可报告的真实 Accuracy/AUROC/AUPR/AURC、GPU 显存或耗时数据。

已执行 `python -m unittest Classifier.test_stage1 -v`，7 项全部通过，包括损失梯度方向、4271 维 BCE reduction、L3 无配对边界、四组 optimizer step、模拟权重兼容、LoRA 冻结、平均池化前向一致、断点恢复后第二个 epoch 参数与连续训练逐位相同、完整 logits/Sigmoid 导出一致、重叠划分拒绝。语法编译、CLI --help 和 git diff --check 通过。

测试环境：现有 Python 3.8.20 / torch 2.4.1+cpu，测试依赖隔离放在 Classifier/.test_deps（timm 1.0.22、sklearn 1.3.2 等），未修改已有 Python 环境安装。最初网络沙箱与旧 Python TLS 阻止下载，随后使用新版 Python 下载官方 PyPI wheel、离线装入工作区。合成测试发现并修复了新版 timm norm_layer 接收 device/dtype 参数的兼容问题。正式服务器建议独立 Python 3.10/3.11 环境和匹配的 CUDA torch/torchvision。

### 产物约定与已知限制

- checkpoint 保存 LoRA+head，不重复存冻结骨干；恢复/导出必须提供 SHA256 相同的 PlantCLEF 原权重。
- 导出 logits.npy 与 sigmoid.npy 均为 N×4271 float32；samples.csv 逐行对应样本、标签、预测、正确标记、最大分数、top1/top2 与真实/错误类 margin。另有 per_class、risk_coverage、metrics 和 provenance。
- 预测用 argmax(logits) 避免浮点 Sigmoid 饱和造成伪并列；Sigmoid 分数不直接解释为正确概率。
- 数据审计检测同名、同 inode 和软链接交叉，未做全量内容去重或 observation 身份隔离；需要原始元数据再核实。可用 audit --verify-images 解码全部图像并统计尺寸。
- 未在真实权重上验证形状/键匹配或测量 GPU 显存；真实权重不匹配时明确报错，不能静默放宽加载。位置编码不插值，固定 224px。
- tau=0.5 仅作描述性统计；频率分组使用 classifier_train 的 >100 / 20–100 / <20，空组返回 null。BCE 正负不平衡可能影响收敛，须观察日志后另开实验。
- outputs、权重、预测矩阵和本机依赖已被忽略。未自动 git commit 或 push；外部 Hand_off.md 不在当前仓库。

### 下一步

1. 在服务器建立独立环境、核实四个路径，执行测试与 audit --verify-images。
2. 用单独的两 epoch smoke 配置验证真实权重与显存，必要时统一调整四组 batch。
3. 正式跑 L1/L2 seed=42，检查诊断数值与梯度量级；再跑组合与另外两个 seeds。
4. 导出每组 best 的 classifier_val，汇总比较并记录选择理由。
5. 固定模型后导出 detector_train、detector_calibration、official_val 进入阶段二；后续每次迭代继续在 Hand_off.md 末尾追加目的、决策、代码、实际结果、问题与下一步。
