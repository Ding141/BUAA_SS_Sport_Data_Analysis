# WISDM 12 类动作识别深度学习项目

本项目面向 WISDM Smartphone and Smartwatch Activity and Biometrics Dataset，使用手机加速度计和陀螺仪数据完成 12 类人体动作识别。项目重点不是追求随机划分下的表面高准确率，而是在更接近真实应用的跨受试者测试场景中，选择准确率、泛化能力、可解释性和复现性更均衡的深度学习模型。

## 一、项目评估原则

本项目采用固定受试者划分，而不是随机窗口划分：

- 训练集：subject `<= 1639`
- 验证集：subject `1640-1644`
- 测试集：subject `>= 1645`
- 任务类别：12 类 WISDM 动作（原18类剔除 soup/chips/drinking/typing/clapping/dribbling 六个极低准确率类别）

这种设置要求模型识别未见过受试者的动作，更能反映真实部署中的泛化能力。

## 二、最终推荐模型

最终推荐方案是 `FeatureMLP + FeatureFusionNet` 软投票集成，输入仍然是手机加速度计和陀螺仪融合后的 182 维 ARFF 时频统计特征。

选择该方案并不是单纯因为测试准确率最高，而是综合考虑了以下因素：

- **跨受试者泛化更稳**：在未见过受试者上，准确率高于原始序列模型。
- **特征含义清楚**：输入包含均值、峰值、方差、绝对偏差、MFCC 等可解释特征。
- **结构互补**：`FeatureMLP` 负责整体特征建模，`FeatureFusionNet` 显式区分加速度计和陀螺仪分支，二者错误模式不完全一致。
- **复杂度受控**：更深的残差特征网络单独测试没有带来更好泛化，因此最终采用软投票集成，而不是盲目堆叠更深网络。
- **可解释性可保留**：基础特征模型仍可通过输入梯度分析传感器、轴向和特征族贡献。

当前主要结果如下：

| 模型 | 输入 | 测试准确率 | Macro F1 |
| --- | --- | ---: | ---: |
| FeatureFusionNet + FeatureMLP 软投票 (0.78:0.22) | 182 维双传感器时频特征 | 0.5898 | 0.5362 |
| FeatureFusionNet | 182 维双传感器时频特征 | 0.5750 | 0.5217 |
| FeatureMLP | 182 维双传感器时频特征 | 0.5420 | 0.4844 |
| InceptionTime (final) | 原始 `6 x 200` 传感器窗口 | 0.5058 | 0.5035 |
| FeatureResNet | 182 维双传感器时频特征 | 0.4959 | 0.4490 |
| ResNet1D (val-selected) | 原始 `6 x 200` 传感器窗口 | 0.4821 | 0.4786 |

## 三、目录结构

```text
sign-deep-learning/
  src/
    deep_models.py          原始序列 CNN/RNN/Transformer 模型
    wisdm_data.py           原始 6 通道窗口加载与缓存
    wisdm_arff.py           ARFF 时频特征加载与缓存
  models/wisdm_deep/
    wisdm_feature_mlp_best.pt       最终推荐模型
    fused_arff_features_phone.npz   推荐模型使用的特征缓存
    wisdm_deep_model.pt             原始序列 InceptionTime 基线
    fused_windows_phone.npz         原始序列窗口缓存
  reports/wisdm_deep/
    指标、混淆矩阵、模型选择记录和可解释性结果
    figures/                        可视化图表
  docs/
    project_report.md               中文汇报文档
```

## 四、常用命令

安装依赖：

```bash
pip install -r requirements.txt
```

使用最终推荐模型预测一个缓存样本：

```bash
python predict_feature_ensemble.py --sample 0
```

重新训练最终推荐模型：

```bash
python train_wisdm_feature_mlp.py --epochs 120
```

评估原始序列基线模型：

```bash
python evaluate_wisdm_deep.py --split test
```

生成项目可视化图表：

```bash
python visualize_project_summary.py
```

生成推荐模型可解释性分析：

```bash
python explain_feature_mlp.py
```

评估最终软投票集成：

```bash
python evaluate_feature_ensemble.py --scan-alpha --alpha-step 0.01
```

## 五、主要可视化产物

图表位于 `reports/wisdm_deep/figures/`：

- `unified_project_architecture.png`：统一项目架构图。
- `project_model_selection.png`：模型选择对比。
- `feature_mlp_training_curve.png`：推荐模型训练曲线。
- `feature_mlp_confusion_matrix_normalized.png`：推荐模型归一化混淆矩阵。
- `feature_mlp_per_class_f1.png`：推荐模型每类 F1。
- `feature_mlp_top_feature_importance.png`：推荐模型重要特征。
- `feature_mlp_group_importance.png`：按传感器、轴向和特征族汇总的重要性。

## 六、模型架构说明

`FeatureMLP` 的详细结构、输入特征、训练策略和可解释性分析见：

```text
docs/feature_mlp_architecture.md
```

第二种输入方式下新增的复杂模型包括：

- `FeatureResNet`：残差式特征网络，单模型测试准确率为 0.4959（12类精简后）。
- `FeatureFusionNet`：加速度计和陀螺仪双分支特征融合网络，单模型测试准确率为 0.5750（12类精简后最佳单模型）。
- `FeatureFusionNet + FeatureMLP`：软投票集成（FusionNet:MLP = 0.78:0.22），当前测试准确率为 0.5898，是12类精简后的最佳结果。
