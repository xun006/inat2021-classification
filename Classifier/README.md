# Unified classifier experiments

正式实验只有 A/B/C/D/G/H/I，统一由一个入口运行：

```bash
python -m Classifier.experiment --help
```

代码职责：

- `config.json`：唯一基础配置及 A–D、G–I 两套训练周期；
- `protocol.py`：实验组定义、正式 seeds 和输出路径；
- `model.py`：Random/ImageNet/PlantCLEF × Head/LoRA/Full 模型工厂；
- `engine.py`：唯一训练、验证和导出实现；
- `checkpoint.py`：统一 checkpoint，并兼容历史 A–D checkpoint；
- `experiment.py`：train、smoke、resume、export 命令；
- `audit_runs.py`、`summarize_matrix.py`：完整性审计和多 seed 汇总。

完整协议与服务器命令见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。
