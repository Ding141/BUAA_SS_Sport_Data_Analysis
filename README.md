# WISDM 掩码预测编码器——自监督人体活动识别

基于 data2vec 2.0 / MaskCAE 范式的自监督掩码预测编码器，用于 WISDM 智能手表活动识别（18 类）。核心思路：通过预测被掩码区域的教师表征来学习有判别力的传感器信号编码。

## 架构概览

```
Input (B, 6, 200) watch accel+gyro
        │
[Patch Time-Freq Embedding]  ──→ 20 patch tokens + 1 global_freq token
        │
   ┌────┴────┐
   ▼         ▼
x-encoder  y-encoder (EMA)
(掩码+MASK) (完整信号)
   │         │
   ▼         ▼
Predictor  Teacher Targets (Layer2+Layer4)
   │         │
   └────┬────┘
        ▼
  L_pred (MSE at masked positions)
        +
  L_lmm  (log-magnitude FFT loss)
        +
  L_supcon (supervised contrastive on CLS)
        +
  L_sigreg (variance+entropy regularization)
```

- **x-encoder（学生）**: 掩码位置插入可学习 [MASK] token，全自注意力
- **y-encoder（教师）**: 完整原始信号，EMA 更新，提供多层预测目标
- **Multi-mask (M=4)**: 每个样本生成 4 种不同掩码视图，共享教师目标
- **三种掩码策略**: 通道掩码 (25%) / 多尺度时域块掩码 (50%) / 频域 FFT 掩码 (25%)

## 快速开始

### 环境

```bash
pip install torch numpy
```

### 数据准备

将 WISDM 数据集解压至 `wisdm-dataset/wisdm-dataset/raw/watch/`，目录结构：

```
wisdm-dataset/wisdm-dataset/raw/watch/
├── accel/
│   ├── data_1600_accel_watch.txt
│   ├── data_1601_accel_watch.txt
│   └── ...
└── gyro/
    ├── data_1600_gyro_watch.txt
    ├── data_1601_gyro_watch.txt
    └── ...
```

### 训练

```bash
cd encoder_training
python pretrain_and_eval.py
```

全程约 8-10 分钟（RTX 4070 Laptop），包含预训练 → 保存模型 → 线性探测 → 报告各类别准确率。

### 仅使用预训练编码器

```python
from models.encoder import MaskedEncoder
from config import Config

cfg = Config()
cfg.encoder.embed_dim = 192
cfg.encoder.n_layers = 4
cfg.encoder.n_heads = 6
cfg.encoder.mlp_ratio = 3.0

encoder = MaskedEncoder(cfg.encoder)
encoder.load_state_dict(torch.load('checkpoints/pretrained_v2.pt')['x_encoder'])
encoder.eval()

# 提取 CLS 表征
_, _, cls, _ = encoder(x, mask_matrix=None)  # x: (B, 6, 200)
```

## 结果

| 指标 | 值 |
|------|-----|
| 测试准确率 | 52.56% |
| 预训练时间 | ~8 min |
| 参数量 | 1.92M |

粗粒度动作（步行 91.6%、慢跑 99.3%、运球 90.4%）表现良好，精细进食动作（喝汤 2.8%、喝水 3.6%）仍是主要瓶颈。

## 项目结构

```
encoder_training/
├── pretrain_and_eval.py      # 主入口：预训练 + 线性探测
├── config.py                 # 数据类配置
├── train.py                  # EMA 模型 + cosine LR schedule
├── models/
│   ├── encoder.py            # PatchTimeFreqEmbedding + MaskedEncoder
│   └── predictor.py          # Predictor + LightReconHead + SigRegHead + lmm_loss
├── data/
│   ├── dataset.py            # WISDMWindowDataset
│   └── augment.py            # 多掩码策略 (multi-mask M=4)
├── Architecture_Summary.md   # 完整架构文档（含图表）
└── checkpoints/
    └── pretrained_v2.pt      # 预训练权重
```

详细架构说明见 [Architecture_Summary.md](encoder_training/Architecture_Summary.md)。

## 主要参考文献

- data2vec 2.0: Efficient Self-supervised Learning with Contextualized Target Representations
- MaskCAE: Masked Convolutional Autoencoder for Sensor-based Human Activity Recognition (IEEE JBHI 2024)
- Freq-Aware MAE: Frequency-Aware Masked Autoencoder for Time-Series Data (2025)
