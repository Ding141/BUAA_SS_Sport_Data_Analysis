# 信号与系统大作业：运动数据（传感器）分析与分类

## 1. 项目背景

本项任务源自北京航空航天大学 2026 年春季学期《信号与系统》课程大作业。目标是利用手机传感器（加速度计、陀螺仪）采集的时间序列信号，通过信号处理与机器学习方法实现人体运动状态（如走路、跑步、站立等）的自动识别。

## 2. 核心任务目标 (针对 AI 辅助开发)

基于“信号与系统”课程知识，重点通过**频域分析**提取特征，并使用**决策树**算法完成分类。

### A. 信号频率分析 (核心)

1. **时频转换**：对原始时间序列信号进行分窗处理，使用快速傅里叶变换 (FFT) 或短时傅里叶变换 (STFT) 将时域信号转换为频域信号。
2. **数字特征提取**：
   - 计算频率轴上的特征：主频率 (Peak Frequency)、平均频率 (Mean Frequency)、中位数频率 (Median Frequency)。
   - 计算能量特征：频谱能量分布、谱熵 (Spectral Entropy)。

### B. 动作分类模型

1. **算法选择**：使用 **决策树 (Decision Tree)** 作为分类器。
2. **特征工程**：将提取的频域数字特征与部分时域基础特征（均值、方差）合并作为特征向量。
3. **流程**：数据集划分为训练集与测试集，训练决策树，输出准确率 (Accuracy) 及混淆矩阵。

### C. 可视化演示要求

1. **原始波形图**：展示不同动作（如行走 vs 跑步）在时域上的差异。
2. **频谱对比图**：展示不同动作在频域（幅度谱）上的显著特征点。
3. **特征分布散点图**：展示关键频域特征在不同类别间的区分度。
4. **决策树可视化**：导出决策树的判定逻辑图。

## 3. 数据集参考

- **UCI HAR Dataset**: [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **WISDM Dataset**: [Smartphone and Smartwatch Activity and Biometrics](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)

## 4. 交付物要求

1. **代码**：包含数据预处理、特征提取、模型训练、结果可视化的全流程脚本。
2. **文档**：`requirement.txt` (版本依赖) 与 `Readme.md` (运行说明)。
3. **报告**：包含任务问题、原理分析、算法设计、结果与分析。