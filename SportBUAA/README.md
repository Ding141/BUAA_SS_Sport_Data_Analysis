# SportBUAA 自主采集数据集

**对应大作业报告**：第 5.1 节

## 采集方案

使用手机 APP 在腰部采集加速度计和陀螺仪时序信号。

- **活动类别**（6 类）：laying, sitting, standing, walking, walking_upstairs, walking_downstairs
- **传感器**：加速度计 (m/s²) + 陀螺仪 (rad/s)
- **数据格式**：CSV，四列 `seconds_elapsed, z, y, x`，与 UCI HAR 对齐

## 用途

该数据集**不参与训练**，纯粹用于检验在 UCI HAR 上训练的模型对真实自采数据的泛化表现（跨数据集线下推理）。

## 推理流水线（对应报告第 5.2 节）

1. **数据对齐预处理**：线性插值至 50Hz 统一时间网格，m/s²→g 单位换算，坐标轴翻转对齐
2. **滑动窗口切分**：128 点/窗，步长 64（50% 重叠）
3. **StandardScaler 标准化**：使用训练集拟合参数
4. **模型推理**：FeatureFusionNet 逐窗口前向传播，Softmax 输出 6 类概率

