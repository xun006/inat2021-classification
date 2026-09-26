
---

## 2026-09-26 — Stage1-Fix-02：AMP 非有限梯度处理

问题：服务器已成功加载模型并输出 epoch 1 step 0，随后 clip_grad_norm_(error_if_nonfinite=True) 因非有限梯度退出。可能是 FP16 loss scaling 溢出；现有代码在 GradScaler.step/update 之前直接抛错，无法完成自动缩放回退。没有访问服务器或确认真实 GPU 根因，不宣称所有数值问题已解决。

设计与代码：run.py 新增 optimizer_step，在 unscale 后检查梯度。有限时严格裁剪，Inf/NaN 时不裁剪，由已记录溢出的 GradScaler 跳过整个更新并降低 scale；保持模型参数与 AdamW 状态不被污染。非 AMP 非有限梯度、非有限 loss、连续 20 批溢出或整轮无有效更新仍报错。所有四组共用相同逻辑，20 epoch、batch=32、学习率及损失参数不变。批次不重试，LR 按原数据进度推进，须审查跳步频率，避免把大量跳步解释为公平有效训练。

记录：amp_events.jsonl 保存每次跳步、epoch/step 和 scale 前后值；history 新增跳步数、实际更新数和 scale；恢复训练清理未提交 epoch 的 AMP 事件。README 记录语义与首次训练失败后使用新目录的命令约定。

验证：10 项 CPU 测试全部通过，包括真实 torch.amp.GradScaler(cpu) 下注入 Inf/NaN 后参数和 optimizer 状态不变、scale 减半、下一步恢复更新；非 AMP 路径仍报错；原训练/恢复/导出与 20 轮调度测试通过。git diff --check 通过。没有进行真实 CUDA 训练验收。

下一步：服务器拉取修复，用新输出目录从头启动（失败发生在第 1 轮，尚无完整 epoch checkpoint）。如持续溢出需检查真实数据/权重及 GPU 数值表现；修改精度等训练设置时统一应用到四组实验，并另行记录。汇总时每个 loss/seed 仅保留一个成功运行，排除失败尝试。
