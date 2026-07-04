# Archive: 迭代版本代码

此目录存放开发过程中的迭代版本，非最终交付物。

## 版本日志

| 文件 | 版本 | 日期 | 说明 | 被替代原因 |
|------|------|------|------|-----------|
| `advanced_classifiers.py` | V1 | 2026-05 | KNN/SVM/DT 三分类器对比 (72-D) | 特征维度有限，无特征选择 |
| `advanced_classifiers_v2.py` | V2 | 2026-05 | 增强特征集 186-D + Random Forest | 全量特征中噪声拖累 KNN |
| `advanced_classifiers_v3.py` | V3 | 2026-05 | MI/RF 特征选择 + k 值扫描 | 特征集仍缺少滑动窗峰度 |
| `plot_dynamic_vs_static.py` | V1 | 2026-05 | 动/静态 2×2 对比（简单 mean±std） | 动态动作直接平均导致 ±1σ 带过宽 |
| `plot_dynamic_vs_static_v2.py` | V2 | 2026-05 | 峰值对齐 + 百分位包络改进版 | 分析功能已整合至核心脚本 |

## 版本演进

```
V1 (72-D 基础特征)
│
├── main.py ──────────────────────────→ code/main/main.py (继续保留)
├── advanced_classifiers.py ──────────→ 归档于此
├── plot_dynamic_vs_static.py ────────→ 归档于此
│
▼
V2 (186-D 增强特征)
├── advanced_classifiers_v2.py ───────→ 归档于此
├── plot_dynamic_vs_static_v2.py ─────→ 归档于此
│
▼
V3 (186-D + 特征选择)
├── advanced_classifiers_v3.py ───────→ 归档于此
│
▼
V4 (滑动窗峰度分析)
├── advanced_analysis_v4.py ──────────→ code/main/advanced_analysis_v4.py
│
▼
V5 (258-D + 滑动窗峰度 + 特征选择)
├── advanced_classifiers_v5.py ───────→ code/main/advanced_classifiers_v5.py
├── feature_report.py ────────────────→ code/main/feature_report.py
```

## 存档说明

- 归档文件保留了原始的 SAVE_DIR 路径配置（如 `figures/uci/`, `figures/analysis/`），这些路径指向重构前的位置
- 若要重新运行归档脚本，输出图像的保存路径需手动调整为当前目录结构
- 各文件头部注释完整保留了版本信息和迭代原因
- 运行输出结果（准确率数值）已在 `docs/technical_report.md` 中完整记录
