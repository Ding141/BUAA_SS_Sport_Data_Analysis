# 图像归档索引

本目录包含项目的全部输出图像，按主题分为 5 个子目录。

## 目录结构一览

| 目录 | 图像数 | 来源脚本 | 说明 |
|------|--------|----------|------|
| `频谱与特征分析/` | 11 | `code/main/analysis_uci.py` + 归档脚本 | UCI 频域深度分析（A-I 共 9 类分析） |
| `分类流水线/` | 18 | `code/main/main.py` + 归档脚本 V1-V3 | 决策树分类 8 图 + 进阶对比 10 图 |
| `分类器对比/` | 3 | `code/main/advanced_classifiers_v5.py` + `feature_report.py` | V5 最终版：版本演进图 + MI 扫描 + CSV |
| `频域分析/` | 5 | `code/main/advanced_analysis_v4.py` | 滑动窗时频峰度分析系列 |
| `演示/` | 1 | `code/main/demo_waveforms.py` | 6 动作时域波形样例 |

## 旧版目录（legacy）

以下目录保留原始英文文件名，用于脚本兼容性和历史对照：

| 目录 | 说明 |
|------|------|
| `analysis/` | 原始 analysis_uci.py 输出（英文名），已被 `频谱与特征分析/` 取代 |
| `uci/` | 原始 main.py + V1-V3 脚本输出（英文名），已被 `分类流水线/` 取代 |
| `demo/` | 原始 demo_waveforms.py 输出（英文名），已被 `演示/` 取代 |

## 图片详细清单

详见 `docs/technical_report.md` 第五章。
