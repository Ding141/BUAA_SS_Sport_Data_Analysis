import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置路径
data_path = "./UCI HAR Dataset/UCI HAR Dataset"

print("="*70)
print("时域特征提取 - UCI HAR数据集")
print("="*70)

# ============================================
# 1. 定义特征提取函数
# ============================================

def extract_single_axis_features(signal, axis_name):
    """
    从单轴信号提取时域特征
    参数:
        signal: 一维信号数组 (128个时间步)
        axis_name: 'x', 'y', 'z'
    返回:
        特征字典
    """
    features = {}
    
    # 基本统计量
    features[f'mean_{axis_name}'] = np.mean(signal)
    features[f'std_{axis_name}'] = np.std(signal)
    features[f'var_{axis_name}'] = np.var(signal)
    features[f'max_{axis_name}'] = np.max(signal)
    features[f'min_{axis_name}'] = np.min(signal)
    features[f'pp_{axis_name}'] = np.max(signal) - np.min(signal)
    features[f'rms_{axis_name}'] = np.sqrt(np.mean(signal**2))
    
    # 过零率 (Zero Crossing Rate)
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features[f'zcr_{axis_name}'] = len(zero_crossings) / len(signal)
    
    # 四分位距 (Interquartile Range)
    q75, q25 = np.percentile(signal, [75, 25])
    features[f'iqr_{axis_name}'] = q75 - q25
    
    # 偏度 (Skewness)
    features[f'skew_{axis_name}'] = stats.skew(signal)
    
    # 峰度 (Kurtosis)
    features[f'kurt_{axis_name}'] = stats.kurtosis(signal)
    
    return features


def extract_fused_features(signal_x, signal_y, signal_z):
    """
    提取三轴融合特征
    """
    features = {}
    
    # 三轴标准差的平均值
    features['std_mean'] = np.mean([np.std(signal_x), np.std(signal_y), np.std(signal_z)])
    
    # 信号幅度区域 (Signal Magnitude Area)
    features['sma'] = np.mean(np.abs(signal_x) + np.abs(signal_y) + np.abs(signal_z))
    
    # 合向量均方根
    rms_x = np.sqrt(np.mean(signal_x**2))
    rms_y = np.sqrt(np.mean(signal_y**2))
    rms_z = np.sqrt(np.mean(signal_z**2))
    features['rms_total'] = np.sqrt(rms_x**2 + rms_y**2 + rms_z**2)
    
    # 合向量均值（反映重力方向）
    mean_x = np.mean(signal_x)
    mean_y = np.mean(signal_y)
    mean_z = np.mean(signal_z)
    features['mean_total'] = np.sqrt(mean_x**2 + mean_y**2 + mean_z**2)
    
    return features


def extract_all_time_domain_features(signal_x, signal_y, signal_z):
    """
    提取所有时域特征 (11个单轴特征×3轴 + 4个融合特征 = 37个特征)
    """
    features = {}
    
    # 单轴特征
    features.update(extract_single_axis_features(signal_x, 'x'))
    features.update(extract_single_axis_features(signal_y, 'y'))
    features.update(extract_single_axis_features(signal_z, 'z'))
    
    # 融合特征
    features.update(extract_fused_features(signal_x, signal_y, signal_z))
    
    return features


# ============================================
# 2. 加载原始数据
# ============================================

print("\n📂 1. 加载原始数据...")

inertial_path = os.path.join(data_path, "train", "Inertial Signals")

# 读取所有9个原始信号文件
total_acc_x = pd.read_csv(os.path.join(inertial_path, "total_acc_x_train.txt"), sep='\s+', header=None)
total_acc_y = pd.read_csv(os.path.join(inertial_path, "total_acc_y_train.txt"), sep='\s+', header=None)
total_acc_z = pd.read_csv(os.path.join(inertial_path, "total_acc_z_train.txt"), sep='\s+', header=None)

# 读取标签
y_train = pd.read_csv(os.path.join(data_path, "train", "y_train.txt"), sep='\s+', header=None)
activity_labels = pd.read_csv(os.path.join(data_path, "activity_labels.txt"), sep='\s+', header=None)
activity_map = dict(zip(activity_labels[0], activity_labels[1]))

print(f"   数据加载完成: {len(total_acc_x)} 个样本, 每个样本 {total_acc_x.shape[1]} 个时间步")

# ============================================
# 3. 对所有样本提取特征
# ============================================

print("\n🔧 2. 提取时域特征...")

n_samples = len(total_acc_x)
all_features = []

for i in range(n_samples):
    # 获取三轴信号
    signal_x = total_acc_x.iloc[i].values
    signal_y = total_acc_y.iloc[i].values
    signal_z = total_acc_z.iloc[i].values
    
    # 提取特征
    features = extract_all_time_domain_features(signal_x, signal_y, signal_z)
    features['label'] = y_train.iloc[i, 0]
    all_features.append(features)
    
    # 进度提示
    if (i + 1) % 2000 == 0:
        print(f"   已处理 {i+1}/{n_samples} 样本")

# 转换为DataFrame
df_features = pd.DataFrame(all_features)
print(f"\n   特征提取完成! 共 {df_features.shape[0]} 样本 × {df_features.shape[1]} 特征")
print(f"   特征列表: {list(df_features.columns)[:10]}... (共{len(df_features.columns)-1}个特征)")

# ============================================
# 4. 可视化1：查看特征数据
# ============================================

print("\n📊 3. 特征数据概览...")

# 查看前5个样本的特征值
print("\n   前5个样本的特征值（部分）:")
print(df_features.head(10).round(4).iloc[:, :8].to_string())

# 特征统计信息
print("\n   特征统计信息:")
feature_stats = df_features.drop('label', axis=1).describe()
print(f"   特征均值范围: [{df_features.drop('label', axis=1).mean().min():.4f}, {df_features.drop('label', axis=1).mean().max():.4f}]")
print(f"   特征标准差范围: [{df_features.drop('label', axis=1).std().min():.4f}, {df_features.drop('label', axis=1).std().max():.4f}]")

# ============================================
# 5. 可视化2：不同活动的特征分布（箱线图）
# ============================================

print("\n📈 4. 绘制不同活动的特征分布...")

# 选择几个关键特征进行可视化
key_features = ['std_mean', 'zcr_x', 'pp_x', 'mean_x', 'sma', 'rms_total']
df_features['activity'] = df_features['label'].map(activity_map)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    
    # 按活动分组绘制箱线图
    data_to_plot = [df_features[df_features['label'] == i][feature].values for i in range(1, 7)]
    
    bp = ax.boxplot(data_to_plot, labels=[activity_map[i] for i in range(1, 7)], patch_artist=True)
    
    # 设置颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title(f'{feature} 在不同活动上的分布', fontsize=12)
    ax.set_xlabel('活动类型')
    ax.set_ylabel('特征值')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 6. 可视化3：特征相关性热力图
# ============================================

print("\n🔥 5. 绘制特征相关性热力图...")

# 选择部分特征绘制相关性矩阵
corr_features = ['std_x', 'std_y', 'std_z', 'zcr_x', 'zcr_y', 'zcr_z', 
                 'pp_x', 'pp_y', 'pp_z', 'mean_x', 'sma', 'std_mean']
corr_matrix = df_features[corr_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=0.5)
plt.title('时域特征相关性热力图', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================
# 7. 可视化4：不同活动的特征均值对比（雷达图）
# ============================================

print("\n📡 6. 绘制不同活动的特征雷达图...")

# 计算每个活动的特征均值
activity_means = df_features.groupby('label')[key_features].mean()
activity_means.index = [activity_map[i] for i in activity_means.index]

# 归一化特征值用于雷达图
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
activity_means_normalized = pd.DataFrame(
    scaler.fit_transform(activity_means),
    columns=activity_means.columns,
    index=activity_means.index
)

# 绘制雷达图
angles = np.linspace(0, 2 * np.pi, len(key_features), endpoint=False).tolist()
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, (activity, row) in enumerate(activity_means_normalized.iterrows()):
    values = row.values.tolist()
    values += values[:1]  # 闭合
    ax.plot(angles, values, 'o-', linewidth=2, label=activity, color=colors[idx])
    ax.fill(angles, values, alpha=0.1, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(key_features, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('不同活动的特征模式雷达图 (归一化后)', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.show()

# ============================================
# 8. 保存特征数据
# ============================================

print("\n💾 7. 保存特征数据...")

# 保存为CSV文件
output_path = "./extracted_features.csv"
df_features.to_csv(output_path, index=False)
print(f"   特征数据已保存到: {output_path}")

# ============================================
# 9. 特征总结
# ============================================

print("\n" + "="*70)
print("📋 特征提取总结")
print("="*70)

print(f"""
✅ 提取完成!

特征统计:
   - 总样本数: {len(df_features)}
   - 总特征数: {len(df_features.columns) - 2} (37个时域特征 + 标签 + 活动名称)
   - 特征类型: 11个单轴特征 × 3轴 = 33个 + 4个融合特征 = 37个

特征列表:
   单轴特征 (11个/轴):
      mean, std, var, max, min, pp, rms, zcr, iqr, skew, kurt
   
   融合特征 (4个):
      std_mean, sma, rms_total, mean_total

下一步建议:
   1. 使用这些特征训练分类器 (KNN/SVM/决策树)
   2. 观察哪些特征对不同活动区分能力最强
   3. 与官方561维特征对比分类效果
""")

# 显示特征重要性示例（基于方差分析）
print("\n📊 特征区分能力初步分析（基于类间方差）:")

# 计算每个特征的类间方差（简化指标）
feature_variance_ratio = {}
for col in key_features:
    group_means = df_features.groupby('label')[col].mean()
    group_vars = df_features.groupby('label')[col].var()
    overall_var = df_features[col].var()
    # 类间方差 / 总方差
    between_var = group_means.var()
    feature_variance_ratio[col] = between_var / overall_var if overall_var > 0 else 0

# 排序并显示
sorted_features = sorted(feature_variance_ratio.items(), key=lambda x: x[1], reverse=True)
print("   特征区分能力排名 (基于类间方差/总方差):")
for i, (feat, ratio) in enumerate(sorted_features, 1):
    print(f"      {i}. {feat}: {ratio:.4f}")

print("\n✅ 所有可视化已完成! 关闭图形窗口可继续...")
plt.show()
#时域特征提取代码逻辑
# ============================================
#一、整体流程
#text
#二、代码逐段解析
#python
#def get_features(x_data, y_data, z_data):
#    n = len(x_data)                    # 样本数 = 7352
#    feat = np.zeros((n, 6))            # 创建空特征矩阵
#    
#    for i in range(n):                 # 遍历每个样本
#        sx = x_data.iloc[i].values     # 第i个样本的X轴信号(128个数)
#        sy = y_data.iloc[i].values     # 第i个样本的Y轴信号(128个数)
#        sz = z_data.iloc[i].values     # 第i个样本的Z轴信号(128个数)
#        
        # 计算6个特征...
#        feat[i, 0] = 计算值
        # ...
    
#    return feat
#三、6个特征的具体计算
#序号	特征	代码	逻辑
#1	PeakX	np.max(sx) - np.min(sx)	找X轴最大值和最小值，相减。走路时冲击范围约0.45g，上楼约0.52g
#2	StdMean	np.mean([np.std(sx), np.std(sy), np.std(sz)])	分别算三轴标准差，再平均。动态活动波动大(>0.1)，静态波动小(<0.03)
#3	RMS	sqrt(rms_x² + rms_y² + rms_z²)	先算每轴RMS，再算合向量。反映信号整体能量强度
#4	MeanX	np.mean(sx)	X轴均值。受重力影响，站立约0.80g，躺着约0.75g
#5	SMA	np.mean(np.abs(sx)+np.abs(sy)+np.abs(sz))	三轴绝对值求和再平均。总活动量，上楼最大
#6	ZCR_X	len(zero_crossings) / 128	找X轴信号穿越零点的次数，除以总长度128。反映振荡频率/步频
#四、关键细节
#1. RMS计算分两步：

#python
#rms_x = np.sqrt(np.mean(sx**2))      # X轴RMS
#rms_y = np.sqrt(np.mean(sy**2))      # Y轴RMS
#rms_z = np.sqrt(np.mean(sz**2))      # Z轴RMS
#rms_total = np.sqrt(rms_x**2 + rms_y**2 + rms_z**2)  # 合向量
#2. 过零率计算：

#python
#zc = np.where(np.diff(np.signbit(sx)))[0]  # 找到符号变化的位置
#zcr = len(zc) / 128                         # 变化次数 ÷ 总点数
#signbit(sx)：判断每个数是正还是负

#diff()：相邻元素做差，符号变化处值为±2或0

#np.where()：找到非0的位置，即过零点

#五、时间复杂度
#每个样本：128个时间步 × 3轴 = 384次运算

#总样本：7352个

#总运算量：约 7352 × 384 ≈ 282万次浮点运算

#实际运行时间：< 10秒

#六、输出示例
#python
# X_train 的形状和内容
#print(X_train.shape)  # (7352, 6)
#print(X_train[0])     # [0.45, 0.11, 0.89, 0.80, 1.15, 0.22]
# 对应: [PeakX, StdMean, RMS, MeanX, SMA, ZCR_X]
# ============================================