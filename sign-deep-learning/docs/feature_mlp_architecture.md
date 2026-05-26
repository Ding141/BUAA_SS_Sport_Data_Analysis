# FeatureMLP 模型架构详细说明

## 1. 模型定位

`FeatureMLP` 是本项目第二种输入方式下的核心基础模型。它不是直接输入原始时间序列，而是输入由手机加速度计和手机陀螺仪提取出的 182 维时域、频域统计特征。

模型的核心思想是：先把原始传感器信号转换成更稳定、更可解释的运动特征，再用多层感知机完成 18 类动作分类。这样可以减少原始波形中受试者个人习惯、手机姿态和局部噪声对模型的影响，更适合跨受试者泛化场景。

实现文件：

```text
train_wisdm_feature_mlp.py
```

推荐权重：

```text
models/wisdm_deep/wisdm_feature_mlp_best.pt
```

在当前最终方案中，`FeatureMLP` 还会与双分支特征网络 `FeatureFusionNet` 进行软投票集成：

```text
P_final = 0.70 * P_FeatureMLP + 0.30 * P_FeatureFusionNet
```

集成后测试准确率为 0.3887，Macro F1 为 0.3415，高于单独 `FeatureMLP` 的 0.3783 / 0.3306。

## 2. 输入特征

### 2.1 输入维度

模型输入为一个 182 维向量：

```text
x ∈ R^182
```

每一个样本代表一个动作窗口的双传感器融合特征，来源为：

- 手机加速度计：`accel_x, accel_y, accel_z`
- 手机陀螺仪：`gyro_x, gyro_y, gyro_z`

### 2.2 特征类型

输入特征包含以下几类：

- 原始分段统计特征：如 `accel_X0`、`gyro_Z0` 等。
- 均值特征：如 `accel_XAVG`。
- 峰值特征：如 `accel_YPEAK`、`gyro_ZPEAK`。
- 绝对偏差：如 `accel_YABSOLDEV`。
- 标准差：如 `gyro_XSTANDDEV`。
- 方差：如 `accel_YVAR`。
- MFCC 频域特征：如 `accel_XMFCC0`、`gyro_YMFCC3`。
- 合成幅值特征：如 `accel_RESULTANT`、`gyro_RESULTANT`。

这些特征比单个时间点的原始波形更稳定，能够描述动作的整体强度、波动、周期性和旋转变化。

### 2.3 输入标准化

训练前会使用训练集和验证集的统计量对特征进行标准化：

```text
x_norm = (x - mean_train) / std_train
```

其中 `mean_train` 和 `std_train` 会保存到模型 checkpoint 中。测试和推理时必须使用同一组统计量，避免测试集信息泄漏。

## 3. 网络结构

`FeatureMLP` 是一个四层全连接分类网络，结构如下：

```text
输入: 182
  ↓
Linear(182 → 256)
BatchNorm1d(256)
ReLU
Dropout(p=0.30)
  ↓
Linear(256 → 128)
BatchNorm1d(128)
ReLU
Dropout(p=0.25)
  ↓
Linear(128 → 64)
ReLU
Dropout(p=0.15)
  ↓
Linear(64 → 18)
输出: 18 类 logits
```

对应代码：

```python
self.net = nn.Sequential(
    nn.Linear(n_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.30),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.15),
    nn.Linear(64, n_classes),
)
```

## 4. 各层设计解释

### 4.1 第一层：182 → 256

第一层把 182 维输入映射到 256 维隐藏空间。它的作用是组合不同传感器、不同轴向和不同特征族的信息，例如把加速度峰值、陀螺仪频域特征和 resultant 合成幅值结合起来。

这一层维度适当放大，是为了给模型足够空间学习非线性组合关系。

### 4.2 BatchNorm1d

前两层后都使用 `BatchNorm1d`。它的作用包括：

- 缓解不同特征量纲和分布差异；
- 让训练更稳定；
- 减少模型对某一批数据分布的敏感性；
- 对跨受试者泛化有一定帮助。

### 4.3 ReLU

`ReLU` 提供非线性表达能力，使模型能够学习动作类别和特征组合之间的非线性关系。例如，单独的峰值或频域特征可能不足以判断动作，但多个特征共同出现时可以形成有效判别。

### 4.4 Dropout

模型使用三层 Dropout：

```text
p = 0.30, 0.25, 0.15
```

Dropout 从前到后逐渐减小，原因是：

- 前层特征组合更宽，过拟合风险更高，因此使用更强正则化；
- 后层接近分类输出，需要保留更稳定的判别信息，因此 Dropout 较小；
- 这种设计可以减少模型依赖少数强特征，提高泛化能力。

### 4.5 中间层：256 → 128 → 64

隐藏层逐步压缩特征维度：

```text
256 → 128 → 64
```

这种漏斗形结构有两个作用：

- 逐步提炼判别性表示；
- 限制模型容量，避免在样本有限时过拟合受试者特征。

### 4.6 输出层：64 → 18

最后一层输出 18 个 logits，每个 logit 对应一个 WISDM 动作类别。推理时通过 Softmax 转换为概率：

```text
p(class_i | x) = softmax(logits_i)
```

最终预测概率最高的类别。

## 5. 参数规模

按当前结构估算，模型主要参数来自全连接层：

| 层 | 参数量 |
| --- | ---: |
| Linear(182, 256) | 46,848 |
| BatchNorm1d(256) | 512 |
| Linear(256, 128) | 32,896 |
| BatchNorm1d(128) | 256 |
| Linear(128, 64) | 8,256 |
| Linear(64, 18) | 1,170 |
| 总计 | 约 89,938 |

这个规模明显小于复杂循环网络或 Transformer，适合当前样本量下的跨受试者任务。

## 6. 训练策略

### 6.1 数据划分

模型采用按受试者划分：

- 训练集：subject `<= 1639`
- 验证集：subject `1640-1644`
- 测试集：subject `>= 1645`

最终训练时使用训练集 + 验证集训练，在测试受试者上评估。

### 6.2 损失函数

损失函数为带类别权重的交叉熵：

```python
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

类别权重根据训练数据中的类别样本数计算。这样做是为了缓解类别不均衡，使样本较少的动作类别不会在训练中被忽略。

### 6.3 优化器

优化器使用 `AdamW`：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

选择 `AdamW` 的原因：

- 收敛速度快；
- 对不同尺度特征较友好；
- `weight_decay` 可以提供额外正则化，降低过拟合风险。

### 6.4 学习率调度

学习率调度器使用余弦退火：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

它会让学习率在训练后期逐渐降低，使模型从快速探索逐渐转向稳定收敛。

### 6.5 默认训练参数

```text
epochs = 80
batch_size = 128
learning_rate = 1e-3
weight_decay = 1e-4
```

当前保存的推荐模型使用更长训练得到，报告中的训练记录显示测试准确率为 0.3783，Macro F1 为 0.3306。

## 7. 输出与推理流程

推理流程如下：

1. 读取 182 维 ARFF 特征。
2. 使用 checkpoint 中保存的 `mean` 和 `std` 标准化输入。
3. 输入 `FeatureMLP` 得到 18 类 logits。
4. 使用 Softmax 得到类别概率。
5. 输出概率最高的类别，并可显示 Top-K 候选类别。

推理命令：

```bash
python predict_activity.py --sample 0
```

## 8. 可解释性分析

由于 `FeatureMLP` 的输入特征具有明确含义，因此可以直接解释模型依赖了哪些运动模式。

本项目使用输入梯度归因分析：计算模型输出对输入特征的梯度绝对值，并在测试集上取平均，得到各特征的重要性。

可解释性结果显示，模型重点关注：

- 加速度计 X/Y/Z 轴的原始分段特征；
- 陀螺仪 Y/Z 轴的 MFCC 频域特征；
- 加速度计 X/Y 轴的 MFCC 频域特征；
- 陀螺仪 X/Z 轴的原始分段特征；
- 加速度计和陀螺仪的 resultant 合成幅值；
- 峰值、方差、绝对偏差等强度和波动特征。

这说明模型不是依赖单一传感器或单一轴向，而是同时利用平移加速度、旋转变化和频域节律信息。该行为符合人体动作识别的物理直觉。

相关文件：

```text
reports/wisdm_deep/feature_mlp_explainability.csv
reports/wisdm_deep/feature_mlp_group_importance.csv
reports/wisdm_deep/figures/feature_mlp_top_feature_importance.png
reports/wisdm_deep/figures/feature_mlp_group_importance.png
```

## 9. 泛化能力分析

`FeatureMLP` 相比端到端原始序列模型更适合当前项目，主要原因是它对跨受试者变化更稳健。

原始序列窗口会保留大量局部波形细节，这些细节可能来自：

- 手机摆放姿态；
- 受试者动作幅度；
- 走路或手部动作速度；
- 单个用户的动作习惯；
- 传感器短时噪声。

而时频统计特征会把局部波形压缩成更稳定的运动描述，例如强度、波动、节律和旋转变化。因此，模型更容易学习动作类别本身的共性，而不是记住某些受试者的局部轨迹。

从可解释性结果看，模型同时利用统计特征、频域特征、合成幅值和双传感器信息，这也支持了它具备更好泛化潜力的判断。

## 10. 局限性

`FeatureMLP` 仍然存在局限：

- 它依赖预先提取的 ARFF 特征，不是完全端到端模型。
- 样本数量少于原始窗口缓存。
- 对 soup、chips、drinking、typing、clapping 等细粒度手部动作仍然区分困难。
- 如果传感器位置变化很大，部分统计特征仍可能受影响。
- 当前模型尚未融合腕表数据，手部动作信息仍不充分。

## 11. 后续改进方向

后续可以从以下方向改进：

1. 对齐原始窗口和 ARFF 特征，构建双输入融合网络。
2. 引入腕表数据，提高细粒度手部动作识别能力。
3. 对困难类别使用 focal loss、重采样或类别专门判别头。
4. 使用自监督预训练提升跨受试者表示能力。
5. 对特征重要性结果进行特征筛选，减少冗余输入，提高模型稳定性。

## 12. 第二种输入方式下的复杂模型扩展

为了尝试进一步提高 182 维 ARFF 特征输入方式的准确率，本项目额外实现了两个更复杂的神经网络。

### 12.1 FeatureResNet

`FeatureResNet` 是残差式特征网络，包含更宽的隐藏层和多个残差块。它的设计目标是增强非线性表达能力，让模型学习更复杂的特征组合。

结构特点：

- 输入投影到 384 维隐藏空间；
- 使用多个残差块；
- 每个残差块包含 Linear、BatchNorm、GELU 和 Dropout；
- 使用验证集 Macro F1 做早停选择。

实验结果：

```text
Accuracy: 0.3518
Macro F1: 0.3135
```

该模型在训练和验证上学习能力较强，但测试集未超过 `FeatureMLP`，说明它存在更高的过拟合风险。

### 12.2 FeatureFusionNet

`FeatureFusionNet` 是双分支特征融合网络。它把 182 维特征按前缀拆成两组：

- `accel_` 加速度计特征；
- `gyro_` 陀螺仪特征。

两条分支分别编码后，融合以下四类表示：

```text
a, g, |a - g|, a * g
```

其中：

- `a` 表示加速度计分支编码；
- `g` 表示陀螺仪分支编码；
- `|a - g|` 表示两个传感器表示的差异；
- `a * g` 表示两个传感器的交互。

实验结果：

```text
Accuracy: 0.3610
Macro F1: 0.3034
```

该模型单独使用也没有超过 `FeatureMLP`，但它的错误模式和 `FeatureMLP` 不完全一致，因此适合用于集成。

### 12.3 软投票集成

最终采用的复杂方案不是单独选择更深网络，而是将 `FeatureMLP` 和 `FeatureFusionNet` 做软投票：

```text
P_final = 0.70 * P_FeatureMLP + 0.30 * P_FeatureFusionNet
```

该方案结果为：

```text
Accuracy: 0.3887
Macro F1: 0.3415
```

这说明，在当前数据规模下，更合理的提准方式不是盲目增加单模型深度，而是利用不同结构模型的互补性，在保持 `FeatureMLP` 稳定性的同时，引入双传感器分支模型提供的额外判别信息。
