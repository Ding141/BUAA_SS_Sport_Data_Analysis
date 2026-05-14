# 信号与系统大作业：运动传感器数据频域分析与分类

## 项目概述

基于手机传感器（加速度计、陀螺仪）信号，通过 FFT/STFT 频域分析提取特征，使用决策树实现人体运动状态的自动识别。

支持双数据集：UCI HAR Dataset 和 WISDM Dataset，统一流水线。

## 项目结构

```
signal-and-system/
├── code/                        # 代码
│   ├── main.py                  # 主流水线：双数据集分类
│   └── analysis_uci.py          # UCI 频域深度分析 (8张图)
├── figures/                     # 输出图像
│   ├── analysis/                # UCI 频域分析图 (A-H)
│   ├── uci/                     # UCI 分类流水线图 (1-8)
│   └── wisdm/                   # WISDM 分类流水线图 (1-8)
├── docs/                        # 文档
│   ├── assignment.md            # 大作业说明
│   ├── Readme.md                # 本文件
│   └── requirements.txt         # Python 依赖
├── UCI HAR Dataset/             # UCI HAR 数据集
├── wisdm-dataset/               # WISDM 数据集
└── 20260428 信号与系统 大作业说明.pptx
```

## 环境要求

- Python 3.9+
- 依赖安装：`pip install -r docs/requirements.txt`

## 运行

```bash
# 在项目根目录下运行：
cd signal-and-system

# UCI HAR 频域深度分析
python code/analysis_uci.py

# UCI HAR 分类流水线
python code/main.py uci

# WISDM 分类流水线
python code/main.py wisdm

# 两个数据集都跑
python code/main.py all
```

## 数据集

| 数据集 | 特点 | 类别数 | 采样率 | 传感器通道 |
|--------|------|--------|--------|------------|
| UCI HAR | 预分窗 (128点/窗) | 6 类 | 50 Hz | body_acc(3) + body_gyro(3) |
| WISDM | 原始流式数据需分窗 | 5 类 | 20 Hz | phone accelerometer(3) |

## 技术路线

1. **数据加载与分窗** — UCI 直接加载预分窗惯性信号（6通道）；WISDM 对原始加速度计数据做滑动窗
2. **FFT 频域分析** — 逐窗口逐通道计算单边幅度谱
3. **特征提取** — 每通道 8 频域特征（主频率、平均频率、中位数频率、频谱能量、谱熵、3频带能量）+ 4 时域特征（均值、方差、峰峰值、过零率）
4. **决策树分类** — 5 折交叉验证超参数搜索，Gini 准则
5. **消融实验** — 对比纯时域 / 纯频域 / 融合特征的分类性能

## 输出图像

### figures/analysis/ — UCI 频域深度分析

| 图 | 内容 |
|----|------|
| A_mean_spectrum_body_acc/gyro | 6活动×3轴 平均FFT幅度谱+标准差 | 
| B_overlay_comparison | 6活动频谱叠加对比 |
| C_psd_comparison | Welch 功率谱密度对比 |
| D_stft_comparison | STFT 时频谱 Walk/Sit/Stand/Lay |
| E_band_energy_bars | 分频带能量堆叠柱状图 |
| F_feature_boxplot | 9个关键频域特征箱线图 |
| G_spectral_scatter | 主频 vs 谱熵/能量散点图 |
| H_radar_summary | 6活动频域指纹雷达图 |

### figures/uci/ 和 figures/wisdm/ — 分类流水线

| 图 | 内容 |
|----|------|
| figure1_waveforms | 各类动作原始波形 |
| figure2_stft | STFT 时频谱 |
| figure3_spectrum | FFT 幅度谱对比 |
| figure4_scatter | 频域特征分布散点图矩阵 |
| figure5_tree | 决策树结构 |
| figure6_cm | 混淆矩阵 |
| figure7_importance | 特征重要性排名 |
| figure8_band | 频带能量分布 |
