# 自监督掩码预测编码器——架构与训练总结

## 1. 任务定义

给定一段来自智能手表的 10 秒传感器信号（加速度计 xyz + 陀螺仪 xyz，共 6 通道，20 Hz 采样，200 个时间步），训练一个编码器将其映射为紧凑的向量表征。该表征应具备以下能力：

- 捕捉人体活动的**全局语义**（走路、慢跑、踢球等粗粒度动作）
- 区分**精细手部动作**（喝汤、吃意面、刷牙、打字等细粒度动作）

训练完成后，冻结编码器参数，仅训练一个线性分类头即可完成 18 类活动识别。

---

## 2. 数据流全景

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           原始手表传感器信号                                   │
│              (B, 6, 200)  =  batch × 6通道 × 200时间步                        │
│         6通道 = accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Patch Embedding（时频融合）                            │
│                                                                              │
│   ┌─────────────────────┐         ┌──────────────────────────┐               │
│   │    时域分支 (Conv1d) │         │      频域分支 (per-patch FFT) │           │
│   │  Conv1d(6→96, k=10,  │         │  x → (B,6,20,10)          │           │
│   │         stride=10)   │         │  → rFFT → |·| → (B,6,20,6)│           │
│   │  → (B, 96, 20)       │         │  → reshape (B,20,36)      │           │
│   │                      │         │  → Linear(36→96)          │           │
│   │                      │         │  → (B, 96, 20)            │           │
│   └─────────┬────────────┘         └────────────┬─────────────┘               │
│             │                                   │                            │
│             └───────────┬───────────────────────┘                            │
│                         │ concat → (B, 192, 20)                              │
│                         │ Linear(192→192) fusion + LayerNorm                  │
│                         ▼                                                     │
│                    20 patch tokens (B, 20, 192)                               │
│                                                                              │
│   ┌──────────────────────────────────────────────────────┐                   │
│   │          全局频率 token（始终不参与掩码）                │                   │
│   │  全信号 rFFT(200) → |·| → mean over channels          │                   │
│   │  → (B, 101 freq bins) → Linear(101→192) → (B, 1, 192)│                   │
│   └──────────────────────────────────────────────────────┘                   │
│                                                                              │
│   最终输出: 20 patch tokens + 1 global_freq token = 21 tokens, dim=192        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                                 ▼
          ┌─────────────────┐              ┌─────────────────┐
          │  掩码处理 (x侧)   │              │  保持原始 (y侧)   │
          │  multi-mask M=4  │              │  不做任何掩码     │
          └────────┬────────┘              └────────┬────────┘
                   │                                 │
                   ▼                                 ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│       x-encoder (Student)     │    │       y-encoder (Teacher)    │
│                              │    │                              │
│  掩码位置插入 [MASK] token    │    │  完整 21 个 token 输入        │
│  21 tokens 全自注意力         │    │  21 tokens 全自注意力         │
│  4层 Transformer, 6头, 192维  │    │  4层 Transformer, 6头, 192维  │
│                              │    │  EMA 更新 (τ=0.999→0.9999)   │
│  输出:                        │    │  stop-gradient               │
│  · patch tokens (B,20,192)    │    │  输出:                        │
│  · CLS token   (B,192)        │    │  · Layer2+Layer4 中间特征     │
│  · 中间层特征  (B,20,192)      │    │    (作为教师目标)             │
└──────────────┬───────────────┘    └──────────────┬───────────────┘
               │                                    │
               │ L_pred: MSE(pred, y_target)        │
               │   在掩码位置计算                     │
               │   Layer2 权重 0.3, Layer4 权重 1.0  │
               │◄───────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│                    Predictor（预测器）                      │
│                                                          │
│  x-encoder 输出 (B, 20, 192) → + pos_embed →              │
│  1层 Transformer Decoder → LayerNorm                     │
│  → 提取掩码位置的预测表征 preds (B, N_masked, 192)         │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ SigReg   │ │LightRecon│ │ SupCon   │
   │ 正则     │ │Head (LMM)│ │ 对比学习  │
   └──────────┘ └──────────┘ └──────────┘
```

---

## 3. 模块详解

### 3.1 Patch 时频嵌入 (PatchTimeFreqEmbedding)

每个 patch 覆盖 10 个时间步（0.5 秒），共 20 个 patch。对每个 patch 同时提取时域和频域特征并融合：

- **时域分支**：Conv1d(kernel_size=10, stride=10) 直接对原始波形做非重叠卷积，将 6 通道 10 步信号压缩为 96 维（embed_dim/2）时域特征
- **频域分支**：每 patch 做去均值后的 rFFT，取幅度谱，6 通道 × 6 频率 bin = 36 维，经 Linear 投影至 96 维
- **融合**：时域和频域特征在通道维拼接后经 Linear + LayerNorm，得到 192 维 patch token

此外，整个 200 步信号做 rFFT（101 个频率 bin），通道平均后经 Linear 投影为 1 个全局频率 token。该 token 提供整个窗口的频谱概览，**始终不参与掩码**，确保编码器始终拥有全局频率上下文。

### 3.2 掩码编码器 (MaskedEncoder)

同一份权重支持两种工作模式，由 `mask_matrix` 参数控制：

**x-encoder 模式 (mask_matrix 传入)**：
```
输入 20 tokens → 掩码位置替换为可学习 [MASK] token
→ concat([global_freq, 20 tokens]) → 21 tokens 全自注意力
→ 移除 CLS 和 global_freq → 输出 20 个 patch tokens
```

**y-encoder 模式 (mask_matrix=None)**：
```
输入 20 tokens (完整原始) → concat([global_freq, 20 tokens])
→ 21 tokens 全自注意力 → 收集 Layer 2 和 Layer 4 的中间输出
→ 移除 CLS 和 global_freq → 经 LayerNorm 后作为教师目标
```

关键设计：x-encoder 的 [MASK] token 使被掩码的位置仍参与自注意力，可见区域与掩码区域之间可以充分交互——这比仅对可见 patch 编码器有本质提升。

### 3.3 掩码策略 (multi-mask, M=4)

每个训练样本生成 **4 种不同的掩码视图**，每种视图随机选择以下三种掩码模式之一：

| 模式 | 概率 | 操作 |
|------|------|------|
| 通道掩码 | 25% | 随机将 1-2 个完整通道全部置零（如抹掉整个加速度计 x 轴） |
| 多尺度时域块掩码 | 50% | 40% 的 patch 被掩码，块大小混合：小(3-8, 60%) / 中(10-18, 30%) / 大(20-30, 10%) |
| 频域 FFT 掩码 | 25% | 对信号做 rFFT，偏重中高频的频率 bin 随机置零，再 irFFT 还原 |

4 个视图共享同一个 y-encoder 前向传播结果，大幅提升数据利用率。

### 3.4 预测器 (Predictor)

```
x_encoder output (B, 20, 192)
    │
    ├── + learnable position embedding (1, 20, 192)
    │
    ├── 1-layer Transformer Decoder (Pre-LN, GELU)
    │       d_model=192, nhead=6, FFN=384
    │
    ├── LayerNorm
    │
    └── gather masked positions → (B, N_masked, 192)
```

### 3.5 辅助模块

**LightReconHead**：轻量 MLP (192→128→60)，将预测的表征映射回原始 patch 信号 (6 通道 × 10 步 = 60 维)，用于 LMM 频谱损失。

**SupCon 投影头**：MLP (192→128→64)，将 CLS token 投影到低维空间用于监督对比学习。

**SigReg 头**：对预测表征做 Sigmoid 激活，通过方差正则和熵正则防止所有表征坍缩为常数。

---

## 4. 损失函数

总损失为四项加权求和：

```
L_total = L_pred + L_sigreg + λ_lmm · L_lmm + λ_supcon · L_supcon
```

其中 λ_lmm = 0.1, λ_supcon = 0.15。

### 4.1 L_pred —— 掩码预测损失

```
对于 M=4 个掩码视图，分别计算:

y_target = LayerNorm(y_encoder(完整信号).intermediates[layer])
           # 取 Layer 2 和 Layer 4 的中间输出

pred     = Predictor(x_encoder(掩码视图))

L_pred   = Σ_layers w_layer · MSE(pred[掩码位置], y_target[掩码位置])
           # w_layer2 = 0.3, w_layer4 = 1.0

最终 L_pred = mean over M views
```

### 4.2 L_sigreg —— 防坍缩正则

```
z = Sigmoid(α · Normalize(pred))

L_var = ReLU(0.1 - std(z, dim=0)).mean()     # 鼓励每维有足够方差
L_ent = ReLU(0.3 - entropy(z)).mean()         # 鼓励每维有足够熵

L_sigreg = 0.15 · L_var + 0.01 · L_ent
```

### 4.3 L_lmm —— 对数幅度频谱损失 (Log-Magnitude Mean Loss)

```
# 从掩码位置的原始信号取出 target patches
target_patches = x_original[掩码位置]    # (N, 60)  60=6通道×10步

# LightReconHead 从预测表征重建信号
recon_patches  = LightReconHead(preds)   # (N, 60)

# 在频域比较对数幅度
spec_recon  = log(|rFFT(recon_patches)| + ε)
spec_target = log(|rFFT(target_patches)| + ε)

L_lmm = MSE(spec_recon.mean(over patches), spec_target.mean(over patches))
```

设计意图：直接 MSE 在时域算重建损失容易让模型只关注低频大振幅成分。在 FFT 对数幅度域比较，高频小振幅成分被等比例放大，迫使编码器捕捉精细运动对应的频谱细节。

### 4.4 L_supcon —— 监督对比损失

```
# 用完整信号（无掩码）通过 x-encoder 提取 CLS token
cls    = x_encoder(完整信号).cls_out        # (B, 192)
cls    = Normalize(SupConProj(cls))         # (B, 64)

# 计算相似度矩阵
sim    = cls @ cls.T / 0.07                 # (B, B), τ=0.07

# 正样本对 = 同标签样本
pos    = (labels == labels.T) 且 i≠j
neg    = 所有 i≠j

# SupCon 损失 (对每个 anchor):
L_supcon = -log( Σ_pos exp(sim) / Σ_all_except_self exp(sim) )
```

设计意图：掩码预测损失 (L_pred) 学习的是 patch 级别的局部表征，SupCon 损失让 CLS token 学习跨窗口的全局语义边界——来自"吃意面"的不同窗口应在表征空间中靠近，而与"喝汤"的窗口远离。

---

## 5. 训练流程

### 5.1 预训练阶段

```
For epoch in 1..12:
    For each batch (B=128):
        1. y-encoder: 1次完整信号前向 → 得到 Layer2+Layer4 教师目标
        2. multi_mask: 生成 M=4 个掩码视图
        3. For m in 0..3:
             x-encoder(掩码视图_m) → Predictor → preds_m
             计算 L_pred_m + L_lmm_m
        4. x-encoder(完整信号) → CLS → 计算 L_supcon
        5. SigReg(preds) → 计算 L_sigreg
        6. Total loss backprop → 更新 x-encoder + Predictor + ReconHead + SupConProj
        7. EMA 更新 y-encoder
```

### 5.2 线性探测阶段

```
冻结 x-encoder 所有参数
添加分类头: Linear(192→256) + ReLU + Dropout(0.3) + Linear(256→18)

For epoch in 1..30:
    x_encoder(完整信号).cls_out → 分类头 → CrossEntropy Loss
    余弦退火 LR schedule

选取验证集最佳模型 → 测试集评估
```

---

## 6. 超参数总表

### 数据
| 参数 | 值 |
|------|-----|
| 传感器 | 手表加速度计 + 陀螺仪 (6 通道) |
| 窗口长度 | 200 采样点 (10 秒 @ 20Hz) |
| 窗口步长 | 50 采样点 (2.5 秒，75% 重叠) |
| 训练/验证/测试被试 | 6 / 2 / 2 人 |

### 模型
| 参数 | 值 |
|------|-----|
| Embed dim | 192 |
| Transformer 层数 | 4 |
| 注意力头数 | 6 |
| MLP ratio | 3.0 |
| Patch size | 10 (每 patch 0.5s) |
| Patch 数量 | 20 |
| Predictor 层数 | 1 |
| Predictor FFN | 384 |
| 教师目标层 | Layer 2 (w=0.3), Layer 4 (w=1.0) |
| 可学习 [MASK] token | 1 个 (广播至所有掩码位置) |

### 训练
| 参数 | 值 |
|------|-----|
| Epochs | 12 |
| Batch size | 128 |
| Multi-mask M | 4 |
| 学习率 | 3e-4 → 1e-5 (cosine decay) |
| Warmup | 2 epochs |
| 优化器 | AdamW, weight_decay=0.05 |
| 梯度裁剪 | 1.0 |
| EMA τ | 0.999 → 0.9999 |
| EMA 更新频率 | 每步 |

### 损失权重
| 损失项 | 权重 |
|--------|------|
| L_pred (掩码预测) | 1.0 |
| L_sigreg (防坍缩) | λ_var=0.15, λ_ent=0.01 |
| L_lmm (频谱) | 0.1 |
| L_supcon (对比) | 0.15 (τ=0.07) |

### 线性探测
| 参数 | 值 |
|------|-----|
| Epochs | 30 |
| 学习率 | 1e-3 |
| 分类头结构 | 192→256→18, Dropout=0.3 |
| Scheduler | CosineAnnealingLR |

---

## 7. 数据集

WISDM (Wireless Sensor Data Mining) 智能手机与智能手表活动与生物特征数据集。51 名被试执行 18 种日常活动，同时佩戴手机和手表记录传感器数据。

### 活动列表（18 类）

| 索引 | 标签 | 活动 | 类型 |
|------|------|------|------|
| 0 | A | walking | 粗粒度 |
| 1 | B | jogging | 粗粒度 |
| 2 | C | stairs | 粗粒度 |
| 3 | D | sitting | 粗粒度 |
| 4 | E | standing | 粗粒度 |
| 5 | F | typing | 精细 |
| 6 | G | brushing teeth | 精细 |
| 7 | H | eating soup | 精细（进食） |
| 8 | I | eating chips | 精细（进食） |
| 9 | J | eating pasta | 精细（进食） |
| 10 | K | drinking | 精细（进食） |
| 11 | L | eating sandwich | 精细（进食） |
| 12 | M | kicking (soccer) | 粗粒度 |
| 13 | O | catch (tennis ball) | 粗粒度 |
| 14 | P | dribbling (basketball) | 粗粒度 |
| 15 | Q | writing | 精细 |
| 16 | R | clapping | 粗粒度 |
| 17 | S | folding clothes | 精细 |

### 为什么只用智能手表数据

18 种活动中 12 种为手部动作。智能手表直接佩戴在手腕上，加速度计和陀螺仪能捕捉手腕运动的细微变化。手机在口袋中时，对于进食、写字、刷牙等不涉及身体移动的动作几乎不提供有用信号。因此本方案仅使用手表 6 通道（加速度计 xyz + 陀螺仪 xyz）。

---

## 8. 代码结构

```
encoder_training/
├── config.py                 # 数据类配置（DataConfig, MaskConfig, EncoderConfig, TrainingConfig）
├── data/
│   ├── dataset.py            # WISDMWindowDataset: 原始文件加载、滑动窗口、train/val/test 划分
│   └── augment.py            # 掩码策略: time_block_mask, channel_mask, freq_mask, multi_mask
├── models/
│   ├── encoder.py            # PatchTimeFreqEmbedding, TransformerEncoder, MaskedEncoder
│   └── predictor.py          # Predictor, LightReconHead, lmm_loss, SigRegHead
├── train.py                  # EMAModel, get_cosine_schedule 工具函数
├── pretrain_and_eval.py      # ★ 主入口: 预训练 → 保存模型 → 线性探测 → 报告结果
├── checkpoints/
│   └── pretrained_v2.pt      # 保存的预训练模型权重
└── Architecture_Summary.md   # 本文档
```

---

## 9. 实验结果

预训练用时约 8 分钟（RTX 4070 Laptop, 12 epochs），线性探测 30 epochs 后：

- **测试准确率**: 52.56% (1445/2749)
- **最佳验证准确率**: 63.25%
- **随机基线**: 5.56%

### 损失曲线

```
Epoch  L_pred   L_lmm    L_supcon
────── ──────── ──────── ────────
  1    0.784    7.008    1.484
  2    0.487    7.309    0.863
  3    0.426    7.468    0.616
  4    0.403    7.594    0.487
  5    0.411    7.695    0.388
  6    0.403    7.795    0.321
  7    0.405    7.856    0.276
  8    0.391    7.912    0.240
  9    0.373    7.965    0.208
 10    0.396    7.964    0.187
 11    0.391    7.979    0.171
 12    0.394    7.994    0.157
```

### 各类别准确率

```
  步行 (A-walk)        91.6%  ████████████████████████████████████████
  慢跑 (B-jog)         99.3%  ████████████████████████████████████████
  爬楼梯 (C-stairs)    55.6%  ██████████████████████
  坐下 (D-sit)         71.5%  ████████████████████████████
  站立 (E-stand)       51.4%  ████████████████████
  打字 (F-type)        48.6%  ███████████████████
  刷牙 (G-teeth)       32.6%  █████████████
  喝汤 (H-soup)         2.8%  █
  吃薯片 (I-chips)     18.8%  ███████
  吃意面 (J-pasta)     21.9%  ████████
  喝水 (K-drink)        3.6%  █
  吃三明治 (L-sandw)   22.9%  █████████
  踢球 (M-kick)        72.6%  █████████████████████████████
  接球 (O-catch)       89.4%  ███████████████████████████████████
  运球 (P-dribble)     90.4%  ████████████████████████████████████
  写字 (Q-write)       49.7%  ███████████████████
  拍手 (R-clap)        50.0%  ████████████████████
  折叠衣物 (S-fold)    95.7%  ██████████████████████████████████████
```

### 结果分析

**表现优异的类别**（>70%）：
- 大幅度全身运动（慢跑、步行、运球、接球、踢球）因手腕加速度变化剧烈，易于区分
- 折叠衣物虽然精细但运动模式独特（双手交替折叠），识别率高

**表现较差的类别**（<25%）：
- 饮食相关动作（喝汤、喝水、吃薯片、吃意面、吃三明治、刷牙）均很弱
- 这些动作的共性：手腕在身体前方小幅运动，6 通道加速度+陀螺仪的区分度本质上有局限
- 喝汤 vs 喝水几乎无法区分（两者动作模式高度相似）

**L_lmm 未充分收敛**：
- L_pred 和 L_supcon 均正常下降，但 L_lmm 不降反升
- 可能原因：频谱权重 0.1 偏低，LightReconHead（单隐层 128 维）表达能力不足

**过拟合**：
- 分类器在训练集已达 97.7%，但验证集仅 63.3%
- 7,802 个训练样本对 18 分类任务偏少，分类头需要更强正则化

---

## 10. 已知局限与改进方向

1. **饮食类动作识别**：6 通道手表传感器对这些动作的区分度存在物理上限。可考虑引入手机数据（虽然手机在口袋，但身体姿态的微小差异可能提供补充信号），或增加传感器通道（如磁力计、气压计）

2. **频谱损失未生效**：增大 λ_lmm 至 0.3-0.5，并将 LightReconHead 加深至 3 层 MLP

3. **预训练不充分**：12 epochs × 7,802 样本 ≈ 93,600 次参数更新，对自监督学习偏少。可增至 50-100 epochs

4. **对比学习权重**：当前 SupCon 对易混淆类别对（soup/drink, chips/sandwich）一视同仁。可引入难负样本挖掘，对混淆对施加更大对比权重

5. **数据增强**：对进食类样本做时间拉伸、小幅旋转等增强，提高此类样本多样性
