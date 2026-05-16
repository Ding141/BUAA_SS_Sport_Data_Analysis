# 信号与系统大作业：运动传感器数据频域分析与分类

## 项目概述

基于手机惯性传感器（加速度计、陀螺仪）信号，通过 FFT/STFT 频域分析提取特征，使用决策树、KNN、SVM、随机森林实现人体运动状态的自动识别。

支持双数据集：UCI HAR Dataset 和 WISDM Dataset，统一流水线。

---

## 项目结构

```
signal-and-system/
├── code/                                  # 代码
│   ├── main.py                            # 主流水线：双数据集分类（决策树 + 消融实验）
│   ├── analysis_uci.py                    # UCI 频域深度分析（8 张图 A-H）
│   ├── plot_dynamic_vs_static.py          # V1：动态/静态 2×2 时频对比（简单均值±σ）
│   ├── plot_dynamic_vs_static_v2.py       # V2：峰值对齐 + 百分位包络改进版
│   ├── advanced_classifiers.py            # V1 进阶：KNN/SVM/Decision Tree 对比（72-D）
│   └── advanced_classifiers_v2.py         # V2 进阶：增强特征集 186-D + Random Forest
├── figures/                               # 输出图像
│   ├── analysis/                          # UCI 频域深度分析图（11 张）
│   └── uci/                               # UCI 分类 + 进阶对比图（15 张）
├── docs/                                  # 文档
│   ├── assignment.md                      # 大作业原始说明
│   ├── Readme.md                          # 本文件
│   └── requirements.txt                   # Python 依赖
├── UCI HAR Dataset/                       # UCI HAR 数据集（不提交）
├── wisdm-dataset/                         # WISDM 数据集（不提交）
└── 20260428 信号与系统 大作业说明.pptx      # 作业 PPT
```

---

## 环境要求

- Python 3.9+
- 依赖安装：`pip install -r docs/requirements.txt`

依赖清单：
```
numpy, scipy, scikit-learn, matplotlib, seaborn
```

---

## 运行

```bash
# 在项目根目录下运行

# ── 频域分析 ──
python code/analysis_uci.py                    # UCI 频域深度分析（A-H）
python code/plot_dynamic_vs_static.py          # 动态/静态 2×2 对比 V1
python code/plot_dynamic_vs_static_v2.py       # 动态/静态 2×2 对比 V2（峰值对齐）

# ── 分类流水线 ──
python code/main.py uci                        # UCI 决策树分类
python code/main.py wisdm                      # WISDM 决策树分类
python code/main.py all                        # 两个数据集都跑

# ── 进阶分类器 ──
python code/advanced_classifiers.py            # KNN + SVM + Decision Tree（72-D）
python code/advanced_classifiers_v2.py         # 增强 186-D + Random Forest
```

---

## 数据集

### UCI HAR Dataset

| 属性 | 值 |
|------|-----|
| 采样率 | 50 Hz |
| 窗口长度 | 128 点（2.56 秒） |
| 训练集 | 21 人，7352 窗口 |
| 测试集 | 9 人，2947 窗口 |
| 传感器 | body_acc (x/y/z) + body_gyro (x/y/z)，共 6 通道 |
| 类别 | 6 类：Walk, WalkUpstairs, WalkDownstairs, Sit, Stand, Lay |

手机轴方向（通过重力分析确定）：`X ≈ 垂直（重力轴）`，`Y ≈ 前后`，`Z ≈ 左右`

### WISDM Dataset

| 属性 | 值 |
|------|-----|
| 采样率 | 20 Hz |
| 格式 | 原始流式数据，需滑动窗分窗 |
| 传感器 | phone accelerometer (x/y/z)，3 通道 |
| 类别 | 18 类（实验中取前 5 类主要动作） |

---

## 特征工程

### 基础特征集（72 维）— `main.py` / `advanced_classifiers.py`

每通道 12 特征 × 6 通道 = 72 维。

**频域（8 维/通道）：**

| 特征 | 说明 | 物理意义 |
|------|------|----------|
| PeakFreq | 幅度谱峰值对应频率 | 主导运动频率 |
| MeanFreq | 频谱质心（加权平均频率） | 能量分布中心 |
| MedianFreq | 累计能量 50% 截止频率 | 能量中位数点 |
| SpectralEnergy | 幅度平方和 | 运动强度 |
| SpectralEntropy | 归一化幅度谱的信息熵 | 频谱"纯净度" |
| BandLow | 0–5 Hz 频带能量 | 低频分量（静态/慢走） |
| BandMid | 5–15 Hz 频带能量 | 中频分量（快走/上楼） |
| BandHigh | 15–25 Hz 频带能量 | 高频分量（剧烈动作/噪声） |

**时域（4 维/通道）：**

| 特征 | 说明 |
|------|------|
| Mean | 信号均值 |
| Variance | 信号方差（总能量） |
| Peak-to-Peak | 峰峰值 |
| Zero-Crossing Rate | 过零率（频率粗略估计） |

### 增强特征集（186 维）— `advanced_classifiers_v2.py`

在基础 72 维基础上，每通道新增 19 特征 × 6 通道 = 114 维增量。

**新增频域特征（12 维/通道）：**

| 特征 | 公式/说明 | 分类意义 |
|------|-----------|----------|
| PeakMag | max(mag) | 主频强度 |
| SpectralSpread | √(Σ (f–centroid)²·mag / Σ mag) | 频谱宽度——动态动作带宽更大 |
| SpectralSkewness | 3 阶中心矩 | 频谱偏斜方向 |
| SpectralKurtosis | 4 阶中心矩 | 频谱峰值尖锐度 |
| SpectralFlatness | exp(mean(log(mag))) / mean(mag) | 0=纯音，1=白噪声 |
| SpectralCrest | max(mag) / mean(mag) | 峰均比——动态动作有更突出的主峰 |
| Rolloff75 | 75% 累计能量截止频率 | 能量集中度度量 |
| Rolloff90 | 90% 累计能量截止频率 | 同上，更保守 |
| Rolloff95 | 95% 累计能量截止频率 | 同上 |
| B_0-1Hz | 0–1 Hz 频带能量 | DC 附近——静态动作集中于此 |
| B_1-3Hz | 1–3 Hz 频带能量 | 步态基频（~1.5–2 Hz） |
| B_3-5Hz | 3–5 Hz 频带能量 | 步态谐波 |
| B_5-8Hz | 5–8 Hz 频带能量 | 中频过渡带 |
| B_8-12Hz | 8–12 Hz 频带能量 | 中高频 |
| B_12-18Hz | 12–18 Hz 频带能量 | 高频 |
| B_18-25Hz | 18–25 Hz 频带能量 | 最高频（Nyquist=25 Hz）|

**新增时域特征（7 维/通道）：**

| 特征 | 说明 |
|------|------|
| RMS | √(mean(signal²))，有效值 |
| Skewness | 偏度，信号分布的对称性 |
| Kurtosis | 峰度，冲击成分的强度 |
| IQR | 四分位距 (Q3–Q1)，稳健离散度 |
| Median | 中位数 |
| Max | 最大值 |
| Min | 最小值 |

---

## 分类结果汇总

### 基准：Decision Tree（72-D）

| 数据集 | Accuracy | 备注 |
|--------|----------|------|
| UCI HAR | 77.67% | 21 人训 / 9 人测，6 类 |
| WISDM | 63.71% | 18 类取 5 类，无陀螺仪 |

### UCI HAR 消融实验（特征重要性）

| 特征组合 | Accuracy | Δ |
|----------|----------|---|
| 仅时域（24-D） | 74.79% | — |
| 仅频域（48-D） | 76.26% | +1.47pp |
| 融合（72-D） | 77.67% | +2.88pp |

> 频域特征单独已超越时域，融合后进一步提升。频谱能量是动态/静态分类的最强单特征（相差 200–2000×）。

### UCI HAR 进阶分类器对比

| 分类器 | V1 (72-D) | V2 (186-D) | 提升 |
|--------|-----------|------------|------|
| Decision Tree | 77.67% | 78.69% | +1.02pp |
| KNN (k=5, distance) | 79.13% | 78.15% | −0.98pp |
| SVM (RBF, C=10) | 85.61% | 88.87% | **+3.26pp** |
| **Random Forest** (200 trees) | — | **89.99%** | 🏆 最佳 |

关键发现：
- **Random Forest 89.99%** 为全局最优，200 棵树投票有效抑制过拟合，对 Sitting/Standing 区分度大幅优于单棵决策树
- **SVM + RBF 核**对新增频域形状特征（Flatness、Crest、Spread）特别敏感，提升 3.26pp
- KNN 在高维空间中受"维度灾难"影响，部分噪声特征干扰距离计算，略有下降
- 主要混淆仍集中在 Sitting ↔ Standing（均为静态低能量动作，频谱分布接近）

---

## 输出图像详表

### figures/analysis/ — UCI 频域深度分析

| 文件 | 内容 |
|------|------|
| analysis_A_mean_spectrum_body_acc.png | 6 活动 × 3 轴 body_acc 平均 FFT 幅度谱 ±1σ |
| analysis_A_mean_spectrum_body_gyro.png | 6 活动 × 3 轴 body_gyro 平均 FFT 幅度谱 ±1σ |
| analysis_B_overlay_comparison.png | 6 活动频谱叠加对比（同一轴叠在一张图上） |
| analysis_C_psd_comparison.png | Welch PSD 对比（加窗平均，更平滑） |
| analysis_D_stft_comparison.png | STFT 时频谱（Walk/Sit/Stand/Lay），展示频率随时间变化 |
| analysis_E_band_energy_bars.png | 三频带能量堆叠柱状图（动态 vs 静态能量差异显著） |
| analysis_F_feature_boxplot.png | 9 个关键频域特征箱线图（6 活动分布） |
| analysis_G_spectral_scatter.png | 主频 vs 谱熵/能量 散点图（活动聚类可视化） |
| analysis_H_radar_summary.png | 6 活动频域指纹雷达图 |
| analysis_I_dynamic_vs_static_2x2.png | V1：动态/静态 时域+频域 2×2 对比（简单均值±σ） |
| analysis_I_v2_aligned_comparison.png | V2：峰值对齐 + 百分位包络，叠加样本窗 |

### figures/uci/ — 分类流水线 + 进阶对比

| 文件 | 内容 |
|------|------|
| figure1_waveforms.png | 6 类动作原始波形示例 |
| figure2_stft.png | STFT 时频谱 |
| figure3_spectrum.png | FFT 幅度谱对比 |
| figure4_scatter.png | 频域特征分布散点图矩阵 |
| figure5_tree.png | 决策树结构可视化 |
| figure6_cm.png | 混淆矩阵 |
| figure7_importance.png | 特征重要性排名 |
| figure8_band.png | 频带能量分布 |
| advanced_cm_comparison.png | V1 三分类器（DT/KNN/SVM）混淆矩阵并排 |
| advanced_metrics_bar.png | V1 三分类器 Accuracy/Macro-F1/Weighted-F1 柱状图 |
| advanced_knn_k_selection.png | KNN 不同 k 值准确率曲线（k=1..50） |
| v2_cm_comparison.png | V2 四分类器（DT/RF/KNN/SVM）混淆矩阵 |
| v2_metrics_bar.png | V2 四分类器指标柱状图 |
| v2_feature_importance.png | Random Forest 特征重要性 Top-30 |
| v2_v1_comparison.png | V1 (72-D) vs V2 (186-D) 准确率对比 |

---

## 关键技术决策

### 1. 峰值对齐平均（`plot_dynamic_vs_static_v2.py`）

V1 直接平均导致动态动作的 ±1σ 带过宽——不同窗口捕获不同步态相位（heel-strike、toe-off、swing）。V2 通过 `np.roll` 将每窗主峰值循环平移到窗口中心后再平均，标准差显著收窄，均值曲线更清晰。

### 2. 百分位包络替代 mean±std（频域）

频域幅度谱本身对相位不敏感，但不同窗口的谱形状仍有差异。改用 median + 25th/75th 百分位比 mean±std 更稳健，不受离群值干扰。

### 3. 细粒度频带（7 段 vs 3 段）

原始 3 段（0–5, 5–15, 15–25 Hz）太粗，无法区分步行（~1.5 Hz 主峰）和上楼（~1 Hz + 更高谐波）。7 段分割（0–1, 1–3, 3–5, 5–8, 8–12, 12–18, 18–25 Hz）精确捕捉了这些差异。

### 4. 21/9 人训练/测试划分

严格按照 UCI HAR 官方划分，避免同一人的窗口同时出现在训练和测试集中，保证泛化能力评估的可靠性。

---

## Git 仓库

- Remote: `https://github.com/Ding141/BUAA_SS_Sport_Data_Analysis.git`
- Current branch: `frequency-analysis`
- 数据集 (`UCI HAR Dataset/`, `wisdm-dataset/`) 已加入 `.gitignore`，不提交
