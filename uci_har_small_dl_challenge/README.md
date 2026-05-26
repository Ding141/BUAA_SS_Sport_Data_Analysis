# UCI HAR 少类别深度学习挑战项目

本项目单独面向少类别数据集 `UCI HAR Dataset`，使用手机加速度计与陀螺仪的 9 通道原始窗口完成 6 类人体活动识别。它借鉴了原目录里 CNN1D 从原始传感器序列学习特征的思路，但没有直接复用原有结论：这里使用官方 train/test 划分、按受试者划分验证集，并加入轻量残差卷积与通道注意力来完成多传感器融合挑战。

## 目录结构

```text
uci_har_small_dl_challenge/
  src/
    config.py          标签、通道和默认路径
    data.py            UCI HAR 加载、标准化、样例导出
    model.py           CNNResidualAttention 模型
    train_utils.py     训练、评估工具
  train.py             训练并保存模型
  evaluate.py          官方测试集评估接口
  predict.py           单样本/外部窗口预测接口
  models/              训练后模型和元数据
  reports/             指标、训练曲线数据
  examples/            训练后导出的预测样例
```

默认数据已经放在本项目内部：

```text
uci_har_small_dl_challenge\data\uci\UCI HAR Dataset
```

如需换数据位置，在命令里传入 `--data-root`。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py --epochs 35
```

输出文件：

- `models/uci_har_cnn_attention.pt`
- `models/uci_har_cnn_attention_meta.json`
- `reports/training_history.json`
- `reports/test_metrics.json`
- `examples/*.npy`

## 测试接口

评估官方测试集：

```bash
python evaluate.py
```

预测官方测试集中的一个样本：

```bash
python predict.py --sample-index 0
```

预测外部窗口文件：

```bash
python predict.py --input examples/0_walking_0.npy
```

外部输入支持 `.npy` 或 `.csv`，形状应为 `(9, 128)` 或 `(128, 9)`。通道顺序为：

```text
body_acc_x, body_acc_y, body_acc_z,
body_gyro_x, body_gyro_y, body_gyro_z,
total_acc_x, total_acc_y, total_acc_z
```

## 方法说明

- 数据集类别少：UCI HAR 只有 6 类，适合课程项目中“少类别深度学习”版本。
- 挑战任务：融合加速度计、陀螺仪、总加速度等 9 路信号，直接从原始窗口学习。
- 模型：`CNNResidualAttention`，先用通道注意力估计不同传感器轴的重要性，再用残差 1D CNN 编码时序模式。
- 评估：训练集内部验证按 subject 划分，最终测试使用 UCI HAR 官方测试集，降低随机窗口混合造成的虚高风险。
