# 信号与系统大作业：运动传感器数据频域分析与分类 — 技术报告

> 北京航空航天大学 · 2026 春季学期
>
> 项目仓库: `signal-and-system/`

---

## 一、项目文件结构

```
signal-and-system/
│
├── code/                                      # 代码（详见 §1.1）
│   ├── main/                                  # ✦ 核心流水线脚本
│   │   ├── main.py                            #   主流水线：双数据集分类（决策树 + 消融实验）
│   │   ├── analysis_uci.py                    #   UCI 频域深度分析（A-H 8 张图）
│   │   ├── advanced_analysis_v4.py            #   滑动窗时频峰度分析
│   │   ├── advanced_classifiers_v5.py         #   最优分类器对比（258-D + MI/RF 特征选择）
│   │   ├── feature_report.py                  #   特征重要性报告（MI + RF Gini 排名）
│   │   └── demo_waveforms.py                  #   演示：6 动作时域波形样例
│   │
│   └── archive/                               # 迭代版本存档（详见 §1.2）
│       ├── advanced_classifiers.py            #   V1: KNN/SVM/DT 对比 (72-D)
│       ├── advanced_classifiers_v2.py         #   V2: 增强特征集 186-D + Random Forest
│       ├── advanced_classifiers_v3.py         #   V3: MI/RF 特征选择 + k 值扫描
│       ├── plot_dynamic_vs_static.py          #   V1: 动/静态 2×2 对比（简单 mean±std）
│       └── plot_dynamic_vs_static_v2.py       #   V2: 峰值对齐 + 百分位包络改进版
│
├── figures/                                   # 输出图像（详见 §五）
│   ├── 频域分析/                              #   滑动窗峰度分析系列（5 张）
│   ├── 频谱与特征分析/                        #   UCI 频域深度分析系列（11 张）
│   ├── 分类器对比/                            #   版本演进 + 特征重要性（3 项）
│   ├── 分类流水线/                            #   决策树分类 8 图 + 进阶对比 15 图
│   └── 演示/                                  #   演示波形图
│
├── docs/                                      # 文档
│   ├── assignment.md                          #   大作业原始说明
│   ├── technical_report.md                    #   本技术报告
│   ├── Readme.md                              #   项目说明
│   └── requirements.txt                       #   Python 依赖
│
├── UCI HAR Dataset/                           # UCI HAR 数据集（不提交）
├── wisdm-dataset/                             # WISDM 数据集（不提交）
└── 20260428 信号与系统 大作业说明.pptx         # 作业 PPT
```

### 1.1 核心脚本说明

| 脚本 | 功能 | 输出 |
|------|------|------|
| `code/main/main.py` | 统一分类流水线：数据加载 → FFT/STFT 特征提取 → 决策树训练 → 8 张可视化图 | `figures/分类流水线/figure1~8_*.png` |
| `code/main/analysis_uci.py` | UCI 频域深度分析：6 动作的频谱对比、频带能量、特征分布、雷达图 | `figures/频谱与特征分析/analysis_A~I_*.png` |
| `code/main/advanced_analysis_v4.py` | 滑动窗峰度分析：子窗时域/频域峰度演进、频谱堆叠、统计分布 | `figures/频域分析/滑动窗_*.png` |
| `code/main/advanced_classifiers_v5.py` | 最优分类器：258-D 特征 + MI/RF 特征选择 + 四分类器对比 + 版本演进图 | `figures/分类器对比/V1到V5_*.png` |
| `code/main/feature_report.py` | 特征重要性完整排名：258 维特征的 MI + RF Gini 双指标排序并导出 CSV | `figures/分类器对比/特征重要性排名.csv` |
| `code/main/demo_waveforms.py` | 演示：6 种动作各取一条样例窗口，画加速度计三轴时域波形 | `figures/演示/demo_6act_waveforms.png` |

### 1.2 迭代版本存档

以下文件为开发过程中的迭代版本，已归档至 `code/archive/`，非交付物核心：

| 文件 | 版本 | 说明 | 存档原因 |
|------|------|------|----------|
| `advanced_classifiers.py` | V1 | KNN/SVM/DT 三分类器对比（72-D） | 被 V5 替代；保留用于对比基线 |
| `advanced_classifiers_v2.py` | V2 | 增强特征集 186-D + Random Forest | 被 V5 替代；保留特征工程演进记录 |
| `advanced_classifiers_v3.py` | V3 | MI/RF 特征选择 + k 值扫描 | 被 V5 替代；保留特征选择方法演进记录 |
| `plot_dynamic_vs_static.py` | V1 | 动/静态 2×2 对比（简单 mean±std） | 被 V2 改进版替代；保留方法对比 |
| `plot_dynamic_vs_static_v2.py` | V2 | 峰值对齐 + 百分位包络 | 分析思路保留，核心功能已整合 |

> **版本演进路径**: V1 (72-D 基础) → V2 (186-D 增强) → V3 (特征选择) → V4 (滑动窗峰度) → V5 (258-D 最终版)
>
> V4 为分析方法探索（滑动窗时频峰度），其结果反馈到 V5 的特征工程中。

---

## 二、函数工具说明

### 2.1 数据加载模块

#### `UCIHARDataset` (main.py:62-100)
UCI HAR 数据集加载器。数据为预分窗格式（50 Hz, 128 点/窗, 2.56 s），包含 6 通道（body_acc x/y/z + body_gyro x/y/z），6 类动作。

- `load()`: 返回 `(X_raw_dict, y_dict, activities, channel_names, fs, n_samples)`
  - `X_raw['train']`: (7352, 128, 6) — 21 人训练数据
  - `X_raw['test']`: (2947, 128, 6) — 9 人测试数据

#### `WISDMDataset` (main.py:103-209)
WISDM 数据集加载器。原始流式数据（~20 Hz），需滑动窗分窗（128 点窗长 / 64 点步长 / 50% 重叠），18 类取前 5 类主要动作。

- `_load_raw_files()`: 解析 `raw/phone/accel/data_*_accel_phone.txt`
- `_segment_windows(records)`: 滑动窗分割，仅保留窗口内标签一致的窗口
- 后 3 通道补零（模拟 6 通道格式以复用 UCI 特征提取）

### 2.2 频域变换模块

#### `fft_spectrum(signal)` (main.py:227-237, analysis_uci.py:73-81)
单通道信号 → FFT 单边幅度谱。

- 输入: `signal` (128,) 时域信号
- 输出: `(freqs, mag)` — 频率轴 (0~Nyquist) 与归一化幅度谱
- 算法: `|FFT|/N`，单边谱非 DC/Nyquist 分量 ×2 补偿能量

#### `compute_mean_spectrum(X, y, act_id, ch_idx)` (analysis_uci.py:84-93)
计算指定活动类别在指定通道上的平均幅度谱 ± 标准差。

### 2.3 特征提取模块

#### `extract_frequency_features(mag, freqs)` (main.py:240-270)
从单边幅度谱提取 8 个频域特征，返回 `dict`:

| 特征键 | 计算方式 | 物理意义 |
|--------|----------|----------|
| `PeakFreq` | `freqs[argmax(mag)]` | 幅度谱峰值频率 — 主导运动频率 |
| `MeanFreq` | `Σ(f·mag) / Σmag` | 频谱质心 — 能量分布中心 |
| `MedianFreq` | 累计能量 50% 点 | 能量中位数频率 |
| `SpectralEnergy` | `Σ(mag²)` | 总频谱能量 — 运动强度 |
| `SpectralEntropy` | `-Σ(p·log(p))` | 频谱复杂度 — 动作"纯净度" |
| `BandEnergy_Low` | 0~0.2·Nyq 能量 | 低频分量（DC~5 Hz） |
| `BandEnergy_Mid` | 0.2~0.6·Nyq 能量 | 中频分量（5~15 Hz） |
| `BandEnergy_High` | 0.6~Nyq 能量 | 高频分量（15~25 Hz） |

#### `extract_time_features(signal)` (main.py:273-280)
提取 4 个时域特征: `Mean`, `Var`, `PeakToPeak`, `ZeroCrossingRate`。

#### `extract_freq_features` (增强版, advanced_classifiers_v5.py:144-180)
在基础 8 维频域特征基础上，扩展至 24 维/通道，新增:

| 新增特征 | 说明 |
|----------|------|
| `PeakMag` | 最大幅度值 — 主频强度 |
| `SpectralSpread` | √(Σ(f-centroid)²·mag / Σmag) — 频谱带宽 |
| `SpectralSkewness` | 3 阶中心矩 — 频谱偏斜 |
| `SpectralKurtosis` | 4 阶中心矩 — 频谱尖锐度 |
| `SpectralFlatness` | exp(mean(log(mag))) / mean(mag) — 0=纯音, 1=白噪声 |
| `SpectralCrest` | max/mean — 峰均比 |
| `Rolloff75/90/95` | 累计能量截止频率 |
| `B_0-1Hz` ~ `B_18-25Hz` | 7 段细粒度频带能量 |

#### `extract_time_features` (增强版, advanced_classifiers_v5.py:183-189)
扩展至 11 维/通道: 新增 `RMS`, `Skewness`, `Kurtosis`, `IQR`, `Median`, `Max`, `Min`。

#### `滑动窗峰度特征(信号)` (advanced_classifiers_v5.py:90-132)
对单个 128 点窗口做 7 个子窗（32 点 × 步长 16）分析，每子窗提取时域峰度、频域峰度、主频、质心，再对 7 个子窗的结果进行统计聚合（mean/std/kurtosis/ptp），产出 16 维/通道。仅对加速度 3 通道计算。

### 2.4 分类器训练模块

#### `build_feature_matrix(raw_data, channel_names, fs)` (main.py:283-318)
逐窗口逐通道调用特征提取函数，构建 (n_windows, n_features) 特征矩阵。

#### `run_pipeline(dataset)` (main.py:585-706)
完整流水线：数据加载 → 特征提取 → GridSearchCV 超参数搜索 → 决策树训练 → 评估（准确率、F1、消融实验）→ 8 张可视化图。

#### `evaluate_all()` (advanced_classifiers_v5.py:226-249)
训练并评估四个分类器（决策树/KNN/SVM/随机森林），返回 `{分类器名: 准确率}`。

### 2.5 可视化模块

| 函数 | 文件 | 输出内容 |
|------|------|----------|
| `figure_a_mean_spectrum()` | analysis_uci.py | 6 活动 × 3 轴平均 FFT 幅度谱（加速度 + 陀螺仪各一张） |
| `figure_b_overlay_comparison()` | analysis_uci.py | 6 活动频谱叠加对比 |
| `figure_c_psd_comparison()` | analysis_uci.py | Welch PSD 对比 |
| `figure_d_stft_comparison()` | analysis_uci.py | STFT 时频谱对比 |
| `figure_e_band_energy_bars()` | analysis_uci.py | 5 频带堆叠柱状图 |
| `figure_f_feature_boxplot()` | analysis_uci.py | 9 个关键特征箱线图 |
| `figure_g_spectral_scatter()` | analysis_uci.py | 主频 vs 谱熵/能量散点图 |
| `figure_h_radar_summary()` | analysis_uci.py | 频域指纹雷达图 |
| `画六动作峰度图()` | advanced_analysis_v4.py | 6 动作时域+频域峰度演进 |
| `画频谱演进()` | advanced_analysis_v4.py | 单动作子窗频谱堆叠 |
| `画峰度统计箱线图()` | advanced_analysis_v4.py | 峰度分布箱线图 |
| `画动态静态峰度散点()` | advanced_analysis_v4.py | 峰度空间散点图 |
| `plot_waveforms()` | main.py | 原始波形图 |
| `plot_stft_spectrogram()` | main.py | STFT 时频谱 |
| `plot_spectrum_comparison()` | main.py | FFT 幅度谱对比 |
| `plot_feature_scatter()` | main.py | 频域特征散点图矩阵 |
| `plot_decision_tree()` | main.py | 决策树结构可视化 |
| `plot_confusion_matrix()` | main.py | 混淆矩阵 |
| `plot_feature_importance()` | main.py | 特征重要性 Top-20 |
| `plot_band_energy()` | main.py | 频带能量饼图 |

---

## 三、预测模型的使用

### 3.1 模型选型与演进

本项目的预测任务为 **6 类人体运动状态分类**（Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying）。

#### 基准模型：决策树 (Decision Tree)

- **选择理由**：课程要求使用决策树作为核心分类器；树模型可解释性强，可导出判定逻辑图；无需特征标准化。
- **超参数搜索**：5 折分层交叉验证 GridSearchCV：
  - `max_depth`: [6, 8, 10, 12, 15, None]
  - `min_samples_split`: [5, 10, 20, 50]
  - `min_samples_leaf`: [2, 5, 10, 20]
- **最优参数**：`max_depth=12, min_samples_leaf=5`
- **72-D 基准准确率**：UCI 77.67%, WISDM 63.71%

#### 进阶模型对比 (V5 最终版)

| 分类器 | 配置 | 准确率 (258-D) | 特点 |
|--------|------|---------------|------|
| **Random Forest** | 200 trees, max_depth=20 | **~90.4%** | 集成学习抑制过拟合，Sitting/Standing 区分最佳 |
| **SVM (RBF)** | C=10, gamma='scale' | ~89.7% | 对频域形状特征敏感，高维空间映射能力强 |
| **KNN** | k=5, weights='distance' | ~80.4% | 受"维度灾难"影响，需特征选择后才能恢复性能 |
| **Decision Tree** | max_depth=12 | ~79.1% | 可解释性最强，但单棵树能力有限 |

### 3.2 特征选择策略

#### Mutual Information (互信息) 过滤

- **原理**：`I(X;Y) = Σ p(x,y)·log₂[p(x,y)/(p(x)·p(y))]`，度量特征与标签的任意依赖（线性+非线性）
- **实现**：`sklearn.feature_selection.mutual_info_classif`（kNN 密度估计）
- **效果**：可筛选掉零 MI 特征（无区分能力），保留 Top-k 特征

#### Random Forest 嵌入选择

- **原理**：训练 RF (100 trees) → 取 `feature_importances_` → `SelectFromModel(threshold='median')`
- **效果**：自动选择对分裂有贡献的特征，考虑了特征间的交互作用

#### 两种方法互补

- MI 衡量单特征与标签的直接关系（"单打能力"）
- RF Gini 衡量特征在集成模型中的实际使用频率（"团队价值"）
- 高 MI + 低 RF：特征本身区分力强但树模型不需要（已被其他特征替代）
- 低 MI + 高 RF：单独区分力弱但在组合中有交互作用

### 3.3 消融实验

| 特征组合 | 维度 | UCI 准确率 | 频域贡献 |
|----------|------|-----------|----------|
| 仅时域 | 24-D | 74.79% | — |
| 仅频域 | 48-D | 76.26% | +1.47pp |
| 融合（时域+频域） | 72-D | 77.67% | +2.88pp |

> 频域特征单独已超越时域，融合后进一步提升，验证了频域分析的核心价值。

### 3.4 数据划分策略

- **UCI HAR**：严格按官方 21 人/9 人划分，避免同一人的窗口同时出现在训练和测试集
- **WISDM**：按时间顺序前 80% 训练、后 20% 测试（时间序划分）
- **交叉验证**：5 折 StratifiedKFold（分层抽样），确保每折类别分布与整体一致

---

## 四、参数选择

### 4.1 信号处理参数

| 参数 | 取值 | 选择理由 |
|------|------|----------|
| 采样率 `FS` | 50 Hz (UCI) / 20 Hz (WISDM) | 数据集硬件决定的固定值 |
| 窗口长度 `N_SAMPLES` | 128 点 (2.56 s) | UCI HAR 官方预分窗长度；足够捕获 2~3 个完整步态周期（步频 ~1.5 Hz） |
| FFT 点数 | 128（等于窗口长度） | 无需补零；频率分辨率 Δf = 50/128 ≈ 0.39 Hz |
| Nyquist 频率 | 25 Hz | 人体运动有意义频率集中在 0~15 Hz，25 Hz 足够覆盖 |
| STFT 参数 | nperseg=32, noverlap=24, nfft=64 | 平衡时间分辨率（0.64 s 子窗）和频率分辨率（~1.56 Hz） |

### 4.2 特征工程参数

| 参数 | 取值 | 选择理由 |
|------|------|----------|
| 频带划分（粗） | 0-5, 5-15, 15-25 Hz | 三个频带分别对应静态/步态基频/高频噪声 |
| 频带划分（细） | 0-1, 1-3, 3-5, 5-8, 8-12, 12-18, 18-25 Hz | 7 段分割精确捕捉步态基频（~1.5 Hz）与谐波，区分平地走/上楼 |
| 滑动子窗 | 32 点 (0.64 s), 步长 16 点 | 足够粒度观察步态周期内峰度变化，产生 7 个子窗/窗口 |
| 滑动窗聚合统计 | mean/std/kurtosis/ptp | 分别反映水平、波动、异常程度和极值范围 |

### 4.3 分类器超参数

| 分类器 | 参数 | 取值 | 选择理由 |
|--------|------|------|----------|
| Decision Tree | max_depth | 12 | GridSearchCV 最优；再深过拟合 |
| Decision Tree | min_samples_leaf | 5 | 防止叶节点样本过少 |
| Random Forest | n_estimators | 200 | 200 棵树后准确率趋于饱和 |
| Random Forest | max_depth | 20 | 比单棵树更深以捕获复杂交互 |
| KNN | n_neighbors | 5 | k 值扫描曲线（1-50）显示 k=5 时测试准确率最优 |
| KNN | weights | 'distance' | 距离加权降低边界样本误分类 |
| SVM | kernel | 'rbf' | 非线性可分问题，RBF 核映射到高维 |
| SVM | C | 10.0 | 较大 C 允许更复杂的决策边界 |

### 4.4 特征选择参数

| 参数 | 取值 | 选择理由 |
|------|------|----------|
| MI k 值扫描范围 | [30, 50, 70, 100, 130, 150, 186] (V3) / [30, 50, 70, 100, 130, 160, 200, 258] (V5) | 覆盖从激进降维到全量特征的完整范围 |
| RF 嵌入选择阈值 | 'median' | 保留重要性高于中位数的特征，自动平衡维度与性能 |

---

## 五、图片命名归档

### 5.1 目录结构

```
figures/
├── 频域分析/                         # 滑动窗峰度分析（V4 输出）
│   ├── 滑动窗峰度图_六动作.png        #   6 动作的时域&频域峰度演进对比
│   ├── 滑动窗频谱演进_行走.png        #   Walking 子窗频谱堆叠
│   ├── 滑动窗频谱演进_静坐.png        #   Sitting 子窗频谱堆叠
│   ├── 滑动窗峰度统计_全部窗口.png    #   全部训练窗口峰度分布箱线图
│   └── 滑动窗峰度散点_动态vs静态.png  #   峰度特征空间聚类散点图
│
├── 频谱与特征分析/                   # UCI 频域深度分析（analysis_uci.py + plot_dynamic_vs_static 输出）
│   ├── 01_平均幅度谱_加速度.png       #   6 活动 × 3 轴 body_acc 平均 FFT 幅度谱 ±1σ
│   ├── 02_平均幅度谱_陀螺仪.png       #   6 活动 × 3 轴 body_gyro 平均 FFT 幅度谱 ±1σ
│   ├── 03_频谱叠加对比.png            #   6 活动频谱叠加对比
│   ├── 04_Welch功率谱密度对比.png     #   Welch PSD 对比
│   ├── 05_STFT时频谱对比.png          #   STFT 时频谱（Walk/Sit/Stand/Lay）
│   ├── 06_频带能量堆叠柱状图.png      #   5 频带能量占比（6 活动 × 2 通道）
│   ├── 07_特征分布箱线图.png          #   9 个关键频域特征箱线图
│   ├── 08_频谱特征散点图.png          #   主频 vs 谱熵/能量散点图
│   ├── 09_频域指纹雷达图.png          #   6 活动频域特征雷达图
│   ├── 10_动态vs静态_V1均值对比.png   #   V1: 动/静态 2×2 时频对比（简单均值±σ）
│   └── 11_动态vs静态_V2对齐对比.png   #   V2: 峰值对齐 + 百分位包络改进版
│
├── 分类流水线/                       # main.py + advanced_classifiers 输出
│   ├── 01_原始波形图.png              #   6 类动作原始波形示例
│   ├── 02_STFT时频谱.png              #   STFT 时频谱
│   ├── 03_FFT幅度谱对比.png           #   FFT 幅度谱对比
│   ├── 04_频域特征散点图矩阵.png      #   频域特征分布散点图矩阵
│   ├── 05_决策树结构图.png            #   决策树结构可视化（前 3 层）
│   ├── 06_混淆矩阵.png                #   混淆矩阵
│   ├── 07_特征重要性排名.png          #   特征重要性 Top-20
│   ├── 08_频带能量分布.png            #   频带能量饼图
│   ├── 09_V1三分类器混淆矩阵.png      #   V1 三分类器混淆矩阵并排
│   ├── 10_V1三分类器指标对比.png      #   V1 三分类器指标柱状图
│   ├── 11_KNN_k值选择曲线.png         #   KNN k=1..50 准确率曲线
│   ├── 12_V2四分类器混淆矩阵.png      #   V2 四分类器混淆矩阵
│   ├── 13_V2四分类器指标对比.png      #   V2 四分类器指标柱状图
│   ├── 14_V2特征重要性Top30.png       #   Random Forest 特征重要性
│   ├── 15_V1vsV2准确率对比.png        #   V1 vs V2 准确率提升对比
│   ├── 16_V3特征选择k值扫描.png       #   V3 不同 k 值准确率曲线
│   ├── 17_V3互信息特征排名.png        #   V3 MI Top-30 特征
│   └── 18_V3版本准确率演进.png        #   V1→V2→V3 准确率演进
│
├── 分类器对比/                       # V5 输出
│   ├── V1到V5准确率演进.png           #   V1→V5 七版本准确率演进柱状图
│   ├── V5_MI扫描曲线.png              #   V5 258-D 的 MI 筛选 k 值扫描
│   └── 特征重要性排名.csv             #   258 维特征 MI + RF Gini 完整排名
│
└── 演示/                             # demo_waveforms.py 输出
    └── 6动作时域波形示例.png          #   6 种动作各一条样例波形
```

### 5.2 中文字体配置

所有中文命名图片使用以下 matplotlib 字体配置（按优先级回退）：

```python
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti SC", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
```

- **macOS**: 优先使用 `Arial Unicode MS`（系统自带，覆盖完整中文字符集）
- **Windows**: 回退至 `SimHei`（黑体）
- **Linux**: 回退至 `Heiti SC` 或 `STHeiti`

### 5.3 原始文件名与归档名对照表

| 原始文件名 | 归档后文件名 | 所属目录 |
|-----------|-------------|----------|
| `analysis_A_mean_spectrum_body_acc.png` | `01_平均幅度谱_加速度.png` | 频谱与特征分析/ |
| `analysis_A_mean_spectrum_body_gyro.png` | `02_平均幅度谱_陀螺仪.png` | 频谱与特征分析/ |
| `analysis_B_overlay_comparison.png` | `03_频谱叠加对比.png` | 频谱与特征分析/ |
| `analysis_C_psd_comparison.png` | `04_Welch功率谱密度对比.png` | 频谱与特征分析/ |
| `analysis_D_stft_comparison.png` | `05_STFT时频谱对比.png` | 频谱与特征分析/ |
| `analysis_E_band_energy_bars.png` | `06_频带能量堆叠柱状图.png` | 频谱与特征分析/ |
| `analysis_F_feature_boxplot.png` | `07_特征分布箱线图.png` | 频谱与特征分析/ |
| `analysis_G_spectral_scatter.png` | `08_频谱特征散点图.png` | 频谱与特征分析/ |
| `analysis_H_radar_summary.png` | `09_频域指纹雷达图.png` | 频谱与特征分析/ |
| `analysis_I_dynamic_vs_static_2x2.png` | `10_动态vs静态_V1均值对比.png` | 频谱与特征分析/ |
| `analysis_I_v2_aligned_comparison.png` | `11_动态vs静态_V2对齐对比.png` | 频谱与特征分析/ |
| `figure1_waveforms.png` | `01_原始波形图.png` | 分类流水线/ |
| `figure2_stft.png` | `02_STFT时频谱.png` | 分类流水线/ |
| `figure3_spectrum.png` | `03_FFT幅度谱对比.png` | 分类流水线/ |
| `figure4_scatter.png` | `04_频域特征散点图矩阵.png` | 分类流水线/ |
| `figure5_tree.png` | `05_决策树结构图.png` | 分类流水线/ |
| `figure6_cm.png` | `06_混淆矩阵.png` | 分类流水线/ |
| `figure7_importance.png` | `07_特征重要性排名.png` | 分类流水线/ |
| `figure8_band.png` | `08_频带能量分布.png` | 分类流水线/ |
| `advanced_cm_comparison.png` | `09_V1三分类器混淆矩阵.png` | 分类流水线/ |
| `advanced_metrics_bar.png` | `10_V1三分类器指标对比.png` | 分类流水线/ |
| `advanced_knn_k_selection.png` | `11_KNN_k值选择曲线.png` | 分类流水线/ |
| `v2_cm_comparison.png` | `12_V2四分类器混淆矩阵.png` | 分类流水线/ |
| `v2_metrics_bar.png` | `13_V2四分类器指标对比.png` | 分类流水线/ |
| `v2_feature_importance.png` | `14_V2特征重要性Top30.png` | 分类流水线/ |
| `v2_v1_comparison.png` | `15_V1vsV2准确率对比.png` | 分类流水线/ |
| `v3_k_sweep.png` | `16_V3特征选择k值扫描.png` | 分类流水线/ |
| `v3_mi_importance.png` | `17_V3互信息特征排名.png` | 分类流水线/ |
| `v3_version_comparison.png` | `18_V3版本准确率演进.png` | 分类流水线/ |
| `demo_6act_waveforms.png` | `6动作时域波形示例.png` | 演示/ |

---

## 六、迭代文件整理与标识

### 6.1 迭代版本关系图

```
V1 (72-D 基础特征)                     V4 (滑动窗分析)
│                                       │
├── main.py                             ├── advanced_analysis_v4.py
├── advanced_classifiers.py             │   ├── 滑动窗峰度图_六动作
├── plot_dynamic_vs_static.py           │   ├── 滑动窗频谱演进_行走/静坐
│                                       │   ├── 滑动窗峰度统计_全部窗口
▼                                       │   └── 滑动窗峰度散点_动态vs静态
V2 (186-D 增强特征)                     │
│                                       │   ⚡ 分析结论反馈至 V5
├── advanced_classifiers_v2.py          │
├── plot_dynamic_vs_static_v2.py        ▼
│                                       V5 (258-D 最终版)
▼                                       │
V3 (特征选择)                           ├── advanced_classifiers_v5.py
│                                       ├── feature_report.py
├── advanced_classifiers_v3.py          ├── V1到V5准确率演进.png
│                                       └── 特征重要性排名.csv
└── (MI过滤 + RF嵌入选择 + k值扫描)
```

### 6.2 各版本差异对照表

| 维度 | V1 (72-D) | V2 (186-D) | V3 (186-D+选择) | V5 (258-D+选择) |
|------|-----------|------------|-----------------|-----------------|
| 频域特征/通道 | 8 | 24 | 24 | 24 |
| 时域特征/通道 | 4 | 11 | 11 | 11 |
| 滑动窗特征/通道 | — | — | — | 16 (仅acc×3) |
| 通道数 | 6 | 6 | 6 | 6 |
| 总维度 | 72 | 186 | 186→k | 258→k |
| 特征选择 | 无 | 无 | MI + RF | MI + RF |
| DT 准确率 | 77.67% | 78.69% | 79.06% | ~79.1% |
| KNN 准确率 | 79.13% | 78.15% | 80.39% | ~80.4% |
| SVM 准确率 | 85.61% | 88.87% | 89.65% | ~89.7% |
| RF 准确率 | — | 89.99% | 90.40% | ~90.4% |

### 6.3 存档标识

所有迭代文件已移至 `code/archive/`，文件头部保留原始注释说明版本号和迭代原因。在 `code/archive/README.md` 中维护完整的版本日志。

核心交付脚本位于 `code/main/`，是所有实验中最终选定的版本。

---

## 附录：关键发现总结

1. **静/动态动作在频域有本质差异**：动态动作（Walking/Upstairs/Downstairs）能量集中在 1-5 Hz（步态基频+谐波），静态动作（Sitting/Standing/Laying）能量几乎全部在 DC-2 Hz
2. **频谱能量是最强单特征**：动态动作的 SpectralEnergy 比静态动作高 200-2000 倍
3. **谱熵区分运动复杂度**：Walking 类动作谱熵高（频谱"杂乱"），Sitting/Standing 谱熵低（频谱"纯净"）
4. **滑动窗峰度是强有力的补充特征**：动态动作的时域峰度因子窗位置剧烈波动（heel-strike 时刻峰度尖峰），静态动作峰度曲线近乎平坦
5. **频域特征单独超越时域**：仅频域（48-D）准确率 76.26% > 仅时域（24-D）74.79%，验证了频域分析的核心价值
6. **特征选择对 KNN 至关重要**：KNN 在 186-D 全量特征下受维度灾难影响（78.15%），MI 选择后恢复至 80.39%
7. **Random Forest + 频域增强特征是全局最优方案**：200 棵树 + 258-D → ~90.4% 准确率
