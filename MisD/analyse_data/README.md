# 新版错图分析

本目录用于分析分类器在 `detector_train` 上的预测，不使用分类器已经见过的
`classifier_train`，也不提前查看保留作最终测试的 `official_val`。

分析结论见 [`ANALYSIS_SUMMARY.md`](ANALYSIS_SUMMARY.md)。

运行：

```bash
python MisD/analyse_data/run_error_analysis.py
```

默认输出到 `MisD/analyse_data/results`：

- `index.html`：所有人工审查页面的总入口；
- `sample_review`：高/低置信度 × 错误/正确四组固定随机样本各 50 张，另含两个极端组；
- `confusion_review`：MSP ≥ 0.90、0.95、0.99 的累计混淆对和两侧类别参考图；
- `tables`：类别错误率、全部错误混淆对、错误分类学层级统计；
- `summary.json`：样本数量、筛选规则和复现参数。

协议：`detector_train` 用于探索和训练错误检测器，`detector_calibration` 用于选阈值，
方法完全固定后才在 `official_val` 上进行一次最终评估。
