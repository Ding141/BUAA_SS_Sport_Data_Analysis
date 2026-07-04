# BUAA 信号与系统大作业 — 基于多传感器融合的人体活动识别

**小组成员**：姚昱呈 丁彦君 李彧 刘垭畅  
**课程**：信号与系统（2026 年春）· 北京航空航天大学  
**GitHub**：[Ding141/BUAA_SS_Sport_Data_Analysis](https://github.com/Ding141/BUAA_SS_Sport_Data_Analysis)

---

## 项目概述

本项目以**人体活动识别（Human Activity Recognition, HAR）**为任务载体，系统实践信号与系统课程的核心知识点——离散时间信号采样、DFT/FFT 频谱分析、STFT 时频分析、滤波与信号分离、信号压缩与表征学习——在 UCI HAR 和 WISDM 两个公开数据集上完成完整实验，并自建 SportBUAA 数据集进行线下推理验证。

### 三层递进路线

```
基础路线                  进阶路线                      挑战路线
时域特征 + 传统ML    →    时频融合特征 + 经典ML    →    深度学习 + 自监督学习
6类活动, UCI HAR          6类活动, UCI HAR             12-18类活动, WISDM
准确率: 74%               准确率: 89.2%                 准确率: 52.56% (18类)
```

| 路线 | 核心方法 | 数据集 | 最佳结果 | 对应目录 |
|------|---------|--------|:---:|------|
| 基础 | 6维时域特征 + 决策树 / GMM | UCI HAR (6类) | 74.0% | `time_domain/` |
| 进阶 | FFT/STFT 频域特征 + 四分类器 (RF最优) | UCI HAR (6类) | **89.2%** | `frequency_analysis/` |
| 挑战 | FeatureFusionNet / 自监督掩码编码器 | WISDM (12/18类) | **52.56%** | `deep_learning/` `masked_encoder/` |
| 推理 | 跨数据集线下推理流水线 | SportBUAA (6类) | — | `prediction_demo/` |

---

## 信号与系统课程知识应用

| 课程知识 | 本项目应用 |
|---------|-----------|
| **离散时间信号与采样** | 加速度计/陀螺仪以 50Hz/20Hz 采样，输出 $x[n]$；人体运动有效频率 0-15Hz 满足奈奎斯特采样定理 |
| **DFT/FFT 频谱分析** | 每窗口 128 点 FFT → 主频、频谱质心、谱熵、分段频带能量。频率分辨率 $\Delta f = 50/128 \approx 0.39$ Hz |
| **STFT 时频分析** | 滑动汉宁窗将 1D 信号展开为 2D 时频谱图，揭示非平稳步态周期内不同阶段的频谱差异 |
| **滤波与信号分离** | 0.3Hz 低通巴特沃斯滤波器分离重力分量与身体运动分量（body_acc = total_acc − gravity） |
| **信号压缩与表征** | 自监督掩码编码器——从残缺信号预测完整表征，等价于信号的紧凑无损编码 |

---

## 目录结构

```
BUAA_SS_Sport_Data_Analysis/
├── README.md                               # 本文件
├── requirements.txt                        # Python 依赖
│
├── time_domain/                            # 基础路线：时域分析
│   └── README.md                           #   6维时域特征 + 决策树/GMM
│
├── frequency_analysis/                     # 进阶路线：频域与时频分析
│   ├── main.py                             #   ★ 主流水线：UCI/WISDM + FFT + 决策树
│   ├── advanced_classifiers_v5.py          #   V1-V5特征迭代 + 四分类器对比
│   ├── advanced_analysis_v4.py             #   V4：滑动窗时频峰度分析
│   ├── stft_improved.py                   #   STFT 时频分析
│   ├── analysis_uci.py                    #   UCI 专项分析
│   ├── classifier_showcase.py             #   分类器展示
│   ├── demo_waveforms.py                  #   波形演示
│   ├── feature_report.py                 #   特征分析报告
│   └── archive/                            #   历史版本 (V1-V3)
│
├── deep_learning/                          # 挑战路线：深度学习
│   ├── src/                                #   核心库
│   │   ├── wisdm_data.py                  #     WISDM 数据加载 (phone/watch)
│   │   ├── wisdm_arff.py                 #     ARFF 时频特征处理
│   │   └── deep_models.py                #     模型定义 (CNN/GRU/LSTM/ResNet等)
│   ├── train_wisdm_feature_fusion.py      #   ★ FeatureFusionNet 训练
│   ├── train_wisdm_feature_mlp.py         #   FeatureMLP 训练
│   ├── train_wisdm_feature_resnet.py      #   FeatureResNet 训练
│   ├── train_wisdm_deep.py               #   端到端模型训练
│   ├── train_wisdm_final.py              #   最终模型训练
│   ├── compare_wisdm_deep_models.py       #   11种模型系统对比
│   ├── evaluate_feature_ensemble.py       #   集成评估
│   ├── evaluate_wisdm_deep.py            #   深度模型评估
│   ├── predict_activity.py               #   活动预测
│   ├── predict_feature_ensemble.py       #   集成预测
│   ├── explain_feature_mlp.py            #   特征可解释性分析
│   ├── visualize_*.py                    #   可视化 (结果/集成/项目总结)
│   ├── project_report.md                  #   深度学习项目汇报
│   ├── models/wisdm_deep/                #   训练好的模型权重
│   └── reports/wisdm_deep/               #   评估报告 + 图表
│
├── masked_encoder/                        # 挑战路线：自监督编码器
│   └── README.md                           #   架构说明 + 18类结果
│
├── prediction_demo/                        # 线下推理流水线 ★可运行
│   ├── predict.py                         #   SportBUAA 推理主入口
│   ├── features.py                        #   26维/通道 时频特征提取
│   ├── model.py                           #   FeatureFusionNet 模型定义
│   └── README.md
│
├── showcase/                               # 展示专项 ★可运行
│   ├── pipeline.py                        #   时域→STFT→FFT→决策树 全流程
│   └── README.md
│
├── docs/                                   # 补充文档
│   ├── assignment.md                      #   作业要求说明
│   └── technical_report.md               #   技术报告
│
└── SportBUAA/                              # 自采数据集
    └── data-for-pr/README.md              #   采集方案与推理说明
```

---

## 快速开始

### 1. 环境配置

```bash
git clone https://github.com/Ding141/BUAA_SS_Sport_Data_Analysis.git
cd BUAA_SS_Sport_Data_Analysis
pip install -r requirements.txt
```

### 2. 数据准备

下载以下数据集并解压至项目根目录：

- **UCI HAR Dataset**：[UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) → `UCI HAR Dataset/`
- **WISDM Dataset**：[WISDM Lab](https://www.cis.fordham.edu/wisdm/dataset.php) → `wisdm-dataset/`

### 3. 运行

```bash
# ── 展示专项（推荐先运行） ──
cd showcase
python pipeline.py
# 输出：showcase/figures/ 下 7 张图 + 终端分类报告

# ── 频域分析主流水线 ──
cd frequency_analysis
python main.py uci          # UCI HAR：FFT + 决策树，输出 figures/
python main.py wisdm        # WISDM：FFT + 决策树
python advanced_classifiers_v5.py   # V1-V5特征迭代 + 四分类器对比

# ── 深度学习 ──
cd deep_learning
python train_wisdm_feature_fusion.py     # FeatureFusionNet 训练
python compare_wisdm_deep_models.py      # 11种模型系统对比
python visualize_project_summary.py      # 生成项目总结图表

# ── 线下推理 ──
cd prediction_demo
python predict.py                        # 预测全部 SportBUAA 样本
python predict.py --sample walking0 --plot   # 单个样本 + 时序曲线
```

---

## 核心发现

1. **频域特征是最关键的单步提升**：纯时域 (~75%) → 时频融合 (~89%)，提升约 **15 个百分点**。走路主频约 1.8 Hz、上楼约 2.2 Hz——FFT 将时域振荡映射为清晰的频域谱峰
2. **手工特征 + 深度学习 > 端到端深度学习**：在小样本 IMU 数据上，手工特征自带降噪能力。FeatureFusionNet (57.5 F1) 显著优于端到端的 InceptionTime (47.9) 和 ResNet1D (44.4)
3. **双分支融合优于简单拼接**：加速度计/陀螺仪分开编码 + 四路融合 (concat + |a-g| + a⊙g) 比直接拼接高 3.3 个百分点
4. **自监督学习突破标签瓶颈**：192 万参数、8 分钟预训练，18 类准确率 52.56%，超过所有监督模型约 14 个百分点
5. **活动粒度决定识别难度**：粗粒度全身运动 >90%（慢跑 99.3%），精细进食动作 <5%——反映传感器物理信息天花板

---

## 数据集

| 数据集 | 类别数 | 采样率 | 窗口长度 | 传感器通道 | 受试者划分 |
|--------|--------|--------|----------|------------|------------|
| UCI HAR | 6 | 50 Hz | 128点, 2.56s | 6通道 (acc+gyro) | 21人训练 / 9人测试 |
| WISDM | 18 | 20 Hz | 200点, 10s | 6通道 (acc+gyro) | 按受试者编号划分 |
| SportBUAA | 6 | ~50 Hz | 128点 | 6通道 | 自主采集，不参与训练 |

---

## 参考文献

[1] Anguita D, et al. A Public Domain Dataset for Human Activity Recognition Using Smartphones. *ESANN 2013*.  
[2] Kwapisz J R, et al. Activity Recognition Using Cell Phone Accelerometers. *ACM SIGKDD 2011*.  
[3] Baevski A, et al. data2vec 2.0: Efficient Self-supervised Learning with Contextualized Target Representations. *arXiv 2022*.  
[4] Liu Z, et al. MaskCAE: Masked Convolutional Autoencoder for Sensor-based Human Activity Recognition. *JBHI 2024*.
