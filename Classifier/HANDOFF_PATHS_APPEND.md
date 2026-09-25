
---

## 2026-09-25 — Stage1-Config-01：确认服务器数据与模型路径

目的：落实用户提供的服务器地址，使运行命令直接对应实际配置。

设计决策与代码变更：核对 Classifier/config.json，发现默认路径已经与用户本次确认一致，因此不制造无效的配置或训练逻辑改动。数据根目录为 /mnt/hdd8t/Mingle/xyyy/MisD/data；当前配置使用其下 classifier_train、classifier_val、class_to_idx.json。预训练模型为 /mnt/hdd8t/Mingle/xyyy/models/misclassification-aware/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth。用户消息中的反斜杠下划线按 Markdown 转义处理，不写入实际文件名。README 的验证集导出占位路径已替换成上述地址，并更新默认路径确认说明。

运行结果：本地检查配置 JSON、数据子路径和模型路径与用户提供的根目录/文件地址一致；没有连接服务器，尚未检查实际子目录存在性。此改动仅涉及运行文档和交接记录，不重跑模型训练测试。

问题：数据根目录下 classifier_train、classifier_val 和 class_to_idx.json 的实际存在性仍由服务器 audit 验证；未取得新的真实训练结果。

下一步：提交并推送本次路径确认文档；服务器拉取 main 后，可直接使用 Classifier/config.json 运行 audit，再进行小规模试跑与正式训练。
