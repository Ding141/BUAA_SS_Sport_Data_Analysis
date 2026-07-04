# 线下推理流水线

**对应大作业报告**：第 5.2-5.3 节

## 文件说明

| 文件 | 功能 |
|------|------|
| `predict.py` | 推理主入口：数据预处理 → 滑窗 → 标准化 → FeatureFusionNet推理 → 时序可视化 |
| `features.py` | 时频域特征提取：每通道 13时域+13频域 → 26维/通道，加速度计3通道→78维，陀螺仪3通道→78维 |
| `model.py` | FeatureFusionNet 双分支融合网络定义 |

## 推理流水线

```
原始CSV (m/s², 不等间隔)
    ↓ 线性插值 → 50Hz均匀网格
    ↓ 单位换算 m/s² → g
    ↓ 坐标轴对齐
预处理后信号
    ↓ 滑动窗口 (128点/窗, 步长64, 50%重叠)
窗口序列 (N, 6, 128)
    ↓ 特征提取 features.py
加速度特征 (N, 78) + 陀螺仪特征 (N, 78)
    ↓ StandardScaler 标准化
    ↓ FeatureFusionNet 推理
Softmax 概率分布 → 6类活动预测
```

## 特征详情 (features.py)

每通道 128 点信号 → 26 维特征：
- **时域 13 维**：mean, std, var, max, min, ptp, rms, zcr, iqr, skew, kurtosis, median, mad
- **频域 13 维**：主频, 频谱质心, 中位数频率, 频谱能量, 谱熵, 5段频带能量(DC-2,2-5,5-10,10-15,15-25Hz), 扩散度, 频谱偏度, 频谱峰度

## 模型架构 (model.py)

```
accel (78-D) → SensorFeatureBranch → a (80-D) ┐
                                               ├→ Fusion(a,g,|a-g|,a⊙g) → 320-D → MLP → 6类
gyro  (78-D) → SensorFeatureBranch → g (80-D) ┘
```

- SensorFeatureBranch：Linear(78→160) + BN + GELU + Dropout(0.25) → Linear(160→80) + BN + GELU + Dropout(0.15)
- 四路融合：concat(a, g, |a-g|, a⊙g) → 320-D
- 分类头：320 → 192 → 96 → 6

## 运行

```bash
cd prediction_demo

# 预测全部样本
python predict.py

# 预测单个样本
python predict.py --sample walking0

# 预测并生成时序曲线图
python predict.py --sample walking0 --plot

# 指定自定义数据目录
python predict.py --data_dir /path/to/SportBUAA/data-for-pr
```

## 依赖

需要由 `deep_learning/train_wisdm_feature_fusion.py` 训练生成的模型权重文件：
- `feature_fusion_har.pt`：模型权重
- `feature_fusion_har_meta.json`：元数据（特征维度、标准化参数）
