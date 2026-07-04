# 挑战路线（一）：多传感器融合深度学习

**对应大作业报告**：第 4.2 节

## 概述

从 UCI HAR 6 类粗粒度活动转向 WISDM 12-18 类精细活动识别。实现并系统对比了 11 种模型，包括端到端 CNN/RNN 和基于手工特征的 MLP 模型。最终推荐方案为 **FeatureMLP + FeatureFusionNet 软投票集成**。

## 目录结构

```
deep_learning/
├── src/                                # 核心库
│   ├── wisdm_data.py                   #   WISDM 数据加载 (phone/watch, 6通道融合)
│   ├── wisdm_arff.py                  #   ARFF 时频特征缓存与加载
│   └── deep_models.py                 #   模型定义 (CNN1D, CNN-GRU/BiGRU, ResNet1D, TCN等)
│
├── train_wisdm_feature_fusion.py       # ★ FeatureFusionNet：双分支融合网络
├── train_wisdm_feature_mlp.py          # FeatureMLP：单分支MLP基线
├── train_wisdm_feature_resnet.py       # FeatureResNet：手工特征+残差网络
├── train_wisdm_deep.py                # 端到端模型训练 (CNN/GRU/LSTM/TCN等)
├── train_wisdm_final.py               # 最终模型训练
│
├── compare_wisdm_deep_models.py        # ★ 11种模型系统对比
├── evaluate_feature_ensemble.py        # 集成评估
├── evaluate_wisdm_deep.py             # 深度模型评估
│
├── predict_activity.py                # 单样本活动预测
├── predict_feature_ensemble.py        # 集成预测
├── explain_feature_mlp.py             # 特征可解释性 (permutation importance)
│
├── visualize_wisdm_results.py         # 训练曲线 + 混淆矩阵 + Per-Class F1
├── visualize_ensemble_results.py      # 集成结果可视化
├── visualize_project_summary.py       # 项目总结图表
│
├── project_report.md                   # 深度学习项目汇报
├── models/wisdm_deep/                 # 训练好的模型权重
└── reports/wisdm_deep/                # 评估报告 + 可视化图表
```

## 快速运行

```bash
cd deep_learning

# 1. 训练 FeatureFusionNet
python train_wisdm_feature_fusion.py

# 2. 训练所有模型并对比
python compare_wisdm_deep_models.py

# 3. 生成项目总结图表
python visualize_project_summary.py
```

## FeatureFusionNet：双分支多传感器融合网络

### 设计动机

加速度计对平移冲击敏感（步态着地），陀螺仪对旋转变化敏感（手腕翻转）。直接拼接 = 默认同质，不合理。应先分别编码，再显式融合。

### 网络架构

```
accel (78-D) ──→ SensorFeatureBranch ──→ a (80-D) ──┐
                                                     ├──→ Fusion(a,g,|a-g|,a⊙g) → MLP → N类
gyro  (78-D) ──→ SensorFeatureBranch ──→ g (80-D) ──┘
```

- **SensorFeatureBranch**：Linear(78→160) + BN + GELU + Dropout → Linear(160→80)
- **四路融合**：$a$ (特异性) | $g$ (特异性) | $|a-g|$ (跨传感器分歧) | $a \odot g$ (共同激活) → 320-D
- **分类头**：320 → 192 → 96 → N_classes

### 12 类活动识别结果 (Macro F1)

| 模型 | Macro F1 | 输入类型 |
|------|:---:|------|
| **FeatureFusionNet** | **57.5** | 182-D 手工特征 |
| Feature Ensemble（集成） | 53.6 | 182-D 手工特征 |
| InceptionTime Final | 50.1 | 原始 6×200 信号 |
| FeatureMLP | 48.4 | 182-D 手工特征 |
| InceptionTime | 47.9 | 原始 6×200 信号 |
| ResNet1D | 44.4 | 原始 6×200 信号 |
| DeepConvLSTM | 40.7 | 原始 6×200 信号 |

FeatureFusionNet 胜出的三个原因：
1. "手工特征 + 深度学习"混合路线——手工特征自带降噪
2. 双分支分离编码比单 MLP 直接拼接高 3.3pp
3. |a-g| 和 a⊙g 显式建模跨传感器交互

### 18 类模型对比 (Accuracy)

| 模型 | Accuracy | 输入类型 |
|------|:---:|------|
| FeatureMLP + FeatureFusionNet 集成 | 38.87% | 182-D 手工特征 |
| FeatureMLP | 37.83% | 182-D 手工特征 |
| FeatureFusionNet | 36.10% | 182-D 手工特征 |
| FeatureResNet | 35.18% | 182-D 手工特征 |
| CNN-LSTM | 31.92% | 原始 6×200 信号 |
| Dual-Branch BiGRU | 29.40% | 原始 6×200 信号 |

详细分析见 `project_report.md`。

## 模型定义 (src/deep_models.py)

支持以下端到端和混合模型架构：

- **CNN1D**：3层 Conv1D + AdaptiveAvgPool
- **CNN-GRU / CNN-BiGRU**：CNN编码器 + GRU时序建模
- **CNN-LSTM**：CNN + LSTM 混合
- **ResNet1D**：1D残差网络
- **TCN**：时序卷积网络
- **InceptionTime**：多尺度卷积
- **DeepConvLSTM**：深度卷积 + LSTM
- **Dual-Branch BiGRU**：双分支双向GRU
- **FeatureMLP**：基于手工特征的MLP
- **FeatureResNet**：手工特征 + 残差MLP
- **FeatureFusionNet**：双分支传感器融合
