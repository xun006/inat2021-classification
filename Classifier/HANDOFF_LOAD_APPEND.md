
---

## 2026-09-25 — Stage1-Fix-01：原始 MAE checkpoint 的 Namespace 兼容

目的与问题：用户服务器训练在加载预训练权重时失败，weights_only=True 拒绝 checkpoint 内的 argparse.Namespace；尚未进入训练。

设计与代码：model.py 使用局部 torch.serialization.safe_globals([Namespace]) 加载原 MAE 的 args 元数据，保留 weights_only=True。test_stage1.py 的模拟 MAE checkpoint 新增真实 Namespace 元数据以覆盖该回归。run.py 在训练加载权重前打印 epochs、batch_size、训练样本量、每 epoch 批次数、loss、seed。

配置说明：每个实验 30 epochs、batch size 32，服务器报告 956507 个训练样本，因此每 epoch 29891 个 batch，末 batch 11 张；每个 epoch 后进行验证。未调整训练参数。

验证：使用含 Namespace 的模拟 checkpoint 运行 7 项 CPU 测试，全部通过。测试发现 torch 2.4 尚无 safe_globals 上下文管理器，已增加 add/get/clear_safe_globals 的临时白名单兼容分支，退出后恢复原白名单；依赖最低版本同步为 torch 2.4 / torchvision 0.19。未访问服务器真实权重或执行完整训练。

下一步：服务器拉取修复后重新执行原训练命令。该错误发生在创建训练输出目录前，通常无需清理输出。若真实文件还有其他未支持对象，需依据实际错误继续检查。
