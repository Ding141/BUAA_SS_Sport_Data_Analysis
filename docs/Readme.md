# 信号与系统大作业：运动传感器数据频域分析与分类

## 项目概述

基于手机惯性传感器（加速度计、陀螺仪）信号，通过 FFT/STFT 频域分析提取特征，使用决策树、KNN、SVM、随机森林实现人体运动状态的自动识别。

支持双数据集：UCI HAR Dataset 和 WISDM Dataset，统一流水线。

---

## 项目结构

```
signal-and-system/
├── code/                                  # 代码
│   ├── main/                              # ✦ 核心交付脚本
│   │   ├── main.py                        #   主流水线：双数据集分类（决策树 + 消融实验）
│   │   ├── analysis_uci.py                #   UCI 频域深度分析（A-H 8 张图）
│   │   ├── advanced_analysis_v4.py        #   滑动窗时频峰度分析（V4）
│   │   ├── advanced_classifiers_v5.py     #   最优分类器对比（V5: 258-D + MI/RF 特征选择）
│   │   ├── feature_report.py              #   特征重要性报告（MI + RF Gini 排名 → CSV）
│   │   └── demo_waveforms.py              #   演示：6 动作时域波形样例
│   │
│   └── archive/                           # 迭代版本存档（详见 archive/README.md）
│       ├── advanced_classifiers.py        #   V1: KNN/SVM/DT 对比 (72-D)
│       ├── advanced_classifiers_v2.py     #   V2: 增强特征集 186-D + Random Forest
│       ├── advanced_classifiers_v3.py     #   V3: MI/RF 特征选择 + k 值扫描
│       ├── plot_dynamic_vs_static.py      #   V1: 动/静态 2×2 对比（简单均值±σ）
│       └── plot_dynamic_vs_static_v2.py   #   V2: 峰值对齐 + 百分位包络改进版
│
├── figures/                               # 输出图像（详见 figures/README.md）
│   ├── 频谱与特征分析/                    #   UCI 频域深度分析系列（11 张）
│   ├── 分类流水线/                        #   决策树分类 + 进阶对比（18 张）
│   ├── 分类器对比/                        #   版本演进 + 特征重要性（3 项，含 CSV）
│   ├── 频域分析/                          #   滑动窗峰度分析系列（5 张）
│   ├── 演示/                              #   演示波形图（1 张）
│   ├── analysis/                          #   [legacy] 原始英文名输出
│   ├── uci/                               #   [legacy] 原始英文名输出
│   └── demo/                              #   [legacy] 原始英文名输出
│
├── docs/                                  # 文档
│   ├── assignment.md                      #   大作业原始说明
│   ├── Readme.md                          #   本文件
│   ├── technical_report.md                #   技术报告（结构、函数、模型、参数、图片归档）
│   ├── implementation_analysis.md         #   基础任务实现分析（代码解析 + 数学原理）
│   └── requirements.txt                   #   Python 依赖
│
├── Makefile                               # 自动化脚本（make help 查看全部命令）
├── UCI HAR Dataset/                       # UCI HAR 数据集（不提交，见 .gitignore）
├── wisdm-dataset/                         # WISDM 数据集（不提交，见 .gitignore）
└── 20260428 信号与系统 大作业说明.pptx     # 作业 PPT
```

---

## 环境要求

- Python 3.9+
- 依赖安装：`pip install -r docs/requirements.txt`

依赖清单：
```
numpy>=1.24.0, scipy>=1.10.0, scikit-learn>=1.2.0, matplotlib>=3.6.0, seaborn>=0.12.0
```

---

## 运行

### 方式一：Makefile（推荐）

```bash
make help          # 查看所有可用命令

make demo          # 6 动作时域波形演示
make analyze       # UCI 频域深度分析（A-I, 11 张图）
make sliding       # 滑动窗时频峰度分析（5 张图）
make classify-uci  # UCI 决策树分类流水线（8 图）
make classify-wisdm  # WISDM 决策树分类流水线
make classify-all  # 双数据集分类流水线
make advanced      # 最优分类器对比（V5, 258-D + MI/RF 特征选择）
make report        # 258 维特征重要性排名 → CSV
make all           # 运行全部脚本
make clean         # 清理输出图像
```

### 方式二：直接运行脚本

```bash
# 在项目根目录下运行

# ── 频域分析 ──
python code/main/analysis_uci.py                    # UCI 频域深度分析（A-H）
python code/main/advanced_analysis_v4.py            # 滑动窗时频峰度分析

# ── 分类流水线 ──
python code/main/main.py uci                        # UCI 决策树分类
python code/main/main.py wisdm                      # WISDM 决策树分类
python code/main/main.py all                        # 两个数据集都跑

# ── 进阶分类器 ──
python code/main/advanced_classifiers_v5.py         # 最优分类器对比 (258-D)

# ── 特征分析 ──
python code/main/feature_report.py                  # 258 维特征重要性排名 → CSV

# ── 演示 ──
python code/main/demo_waveforms.py                  # 6 动作时域波形样例
```

> **注意**: 迭代版本代码位于 `code/archive/`，它们的 SAVE_DIR 配置指向重构前的路径，如需运行请手动调整。

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

### 基础特征集（72 维）— `main.py` / `advanced_classifiers.py`（归档）

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

### 增强特征集（186 维）— `advanced_classifiers_v2.py`（归档）

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

### 最终特征集（258 维）— `advanced_classifiers_v5.py`

在 210 维静态特征（35 特征 × 6 通道）基础上，新增滑动窗峰度特征 48 维（16 特征 × 3 加速度通道）。

**滑动窗特征（16 维/accel 通道）：**
对加速度通道做 7 个子窗（32 点 / 步长 16），每子窗提取时域峰度、频域峰度、主频、质心，然后统计 mean/std/kurtosis/ptp 聚合。

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

### UCI HAR 进阶分类器对比（V5 最终版）

| 分类器 | V1 (72-D) | V2 (186-D) | V3 (186-D+选择) | V5 (258-D+选择) |
|--------|-----------|------------|-----------------|-----------------|
| Decision Tree | 77.67% | 78.69% | 79.06% | ~79.1% |
| KNN (k=5, distance) | 79.13% | 78.15% | 80.39% | ~80.4% |
| SVM (RBF, C=10) | 85.61% | 88.87% | 89.65% | ~89.7% |
| **Random Forest** (200 trees) | — | **89.99%** | **90.40%** | **~90.4%** |

关键发现：
- **Random Forest 90.4%** 为全局最优，200 棵树投票有效抑制过拟合
- **SVM + RBF 核**对新增频域形状特征（Flatness、Crest、Spread）特别敏感
- 特征选择解决了 KNN 在高维空间中的"维度灾难"问题
- 滑动窗峰度特征提供了额外的动/静态区分力

---

## 输出图像详表

### figures/频谱与特征分析/ — UCI 频域深度分析（11 张）

| 文件 | 内容 |
|------|------|
| 01_平均幅度谱_加速度.png | 6 活动 × 3 轴 body_acc 平均 FFT 幅度谱 ±1σ |
| 02_平均幅度谱_陀螺仪.png | 6 活动 × 3 轴 body_gyro 平均 FFT 幅度谱 ±1σ |
| 03_频谱叠加对比.png | 6 活动频谱叠加对比（同一轴叠在一张图上） |
| 04_Welch功率谱密度对比.png | Welch PSD 对比（加窗平均，更平滑） |
| 05_STFT时频谱对比.png | STFT 时频谱（Walk/Sit/Stand/Lay），展示频率随时间变化 |
| 06_频带能量堆叠柱状图.png | 5 频带能量堆叠柱状图（动态 vs 静态能量差异显著） |
| 07_特征分布箱线图.png | 9 个关键频域特征箱线图（6 活动分布） |
| 08_频谱特征散点图.png | 主频 vs 谱熵/能量 散点图（活动聚类可视化） |
| 09_频域指纹雷达图.png | 6 活动频域指纹雷达图 |
| 10_动态vs静态_V1均值对比.png | V1：动态/静态 时域+频域 2×2 对比（简单均值±σ） |
| 11_动态vs静态_V2对齐对比.png | V2：峰值对齐 + 百分位包络，叠加样本窗 |

### figures/分类流水线/ — 分类流水线 + 进阶对比（18 张）

| 文件 | 内容 |
|------|------|
| 01_原始波形图.png | 6 类动作原始波形示例 |
| 02_STFT时频谱.png | STFT 时频谱 |
| 03_FFT幅度谱对比.png | FFT 幅度谱对比 |
| 04_频域特征散点图矩阵.png | 频域特征分布散点图矩阵 |
| 05_决策树结构图.png | 决策树结构可视化 |
| 06_混淆矩阵.png | 混淆矩阵 |
| 07_特征重要性排名.png | 特征重要性排名 |
| 08_频带能量分布.png | 频带能量分布 |
| 09-18_*.png | V1/V2/V3 进阶分类器对比图 |

### figures/频域分析/ — 滑动窗峰度分析（5 张）

| 文件 | 内容 |
|------|------|
| 滑动窗峰度图_六动作.png | 6 动作的时域&频域峰度演进对比 |
| 滑动窗频谱演进_行走.png | Walking 子窗频谱堆叠 |
| 滑动窗频谱演进_静坐.png | Sitting 子窗频谱堆叠 |
| 滑动窗峰度统计_全部窗口.png | 全部训练窗口的峰度分布箱线图 |
| 滑动窗峰度散点_动态vs静态.png | 峰度特征空间动态/静态聚类 |

### figures/分类器对比/ — 版本演进（3 项）

| 文件 | 内容 |
|------|------|
| V1到V5准确率演进.png | V1→V5 七版本准确率演进柱状图 |
| V5_MI扫描曲线.png | V5 258-D 的 MI 筛选 k 值扫描 |
| 特征重要性排名.csv | 258 维特征的 MI + RF Gini 完整排名 |

---

## 关键技术决策

### 1. 峰值对齐平均（`plot_dynamic_vs_static_v2.py`，已归档）

V1 直接平均导致动态动作的 ±1σ 带过宽——不同窗口捕获不同步态相位（heel-strike、toe-off、swing）。V2 通过 `np.roll` 将每窗主峰值循环平移到窗口中心后再平均，标准差显著收窄，均值曲线更清晰。

### 2. 百分位包络替代 mean±std（频域）

频域幅度谱本身对相位不敏感，但不同窗口的谱形状仍有差异。改用 median + 25th/75th 百分位比 mean±std 更稳健，不受离群值干扰。

### 3. 细粒度频带（7 段 vs 3 段）

原始 3 段（0–5, 5–15, 15–25 Hz）太粗，无法区分步行（~1.5 Hz 主峰）和上楼（~1 Hz + 更高谐波）。7 段分割（0–1, 1–3, 3–5, 5–8, 8–12, 12–18, 18–25 Hz）精确捕捉了这些差异。

### 4. 21/9 人训练/测试划分

严格按照 UCI HAR 官方划分，避免同一人的窗口同时出现在训练和测试集中，保证泛化能力评估的可靠性。

### 5. 双指标特征重要性评估

同时使用 Mutual Information（单特征 vs 标签的非线性依赖）和 RF Gini Importance（特征在集成模型中的实际使用频率），两个指标互补。MI 看"单打能力"，RF Gini 看"团队价值"。

---

## 版本演进路线

```
V1 (72-D 基础特征) ──→ V2 (186-D 增强特征) ──→ V3 (186-D + 特征选择)
                                                       │
                                                       ▼
                          V4 (滑动窗峰度分析) ──→ V5 (258-D + 特征选择)
```

详见 `docs/technical_report.md` 第六章和 `code/archive/README.md`。

---

## Git 仓库

- Remote: `https://github.com/Ding141/BUAA_SS_Sport_Data_Analysis.git`
- Current branch: `frequency-analysis`
- 数据集 (`UCI HAR Dataset/`, `wisdm-dataset/`) 已加入 `.gitignore`，不提交
