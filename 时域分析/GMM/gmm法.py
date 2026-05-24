import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import os
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置路径
data_path = "./UCI HAR Dataset/UCI HAR Dataset"

print("="*70)
print("GMM分类器 - 基于时域特征的人体活动识别")
print("="*70)

# ============================================
# 1. 加载数据并提取特征
# ============================================

print("\n📂 1. 加载数据...")

# 加载训练集原始信号
inertial_train = os.path.join(data_path, "train", "Inertial Signals")
train_x = pd.read_csv(os.path.join(inertial_train, "total_acc_x_train.txt"), sep='\s+', header=None)
train_y = pd.read_csv(os.path.join(inertial_train, "total_acc_y_train.txt"), sep='\s+', header=None)
train_z = pd.read_csv(os.path.join(inertial_train, "total_acc_z_train.txt"), sep='\s+', header=None)
y_train = pd.read_csv(os.path.join(data_path, "train", "y_train.txt"), sep='\s+', header=None)

# 加载测试集
inertial_test = os.path.join(data_path, "test", "Inertial Signals")
test_x = pd.read_csv(os.path.join(inertial_test, "total_acc_x_test.txt"), sep='\s+', header=None)
test_y = pd.read_csv(os.path.join(inertial_test, "total_acc_y_test.txt"), sep='\s+', header=None)
test_z = pd.read_csv(os.path.join(inertial_test, "total_acc_z_test.txt"), sep='\s+', header=None)
y_test = pd.read_csv(os.path.join(data_path, "test", "y_test.txt"), sep='\s+', header=None)

# 活动标签映射
activity_labels = pd.read_csv(os.path.join(data_path, "activity_labels.txt"), sep='\s+', header=None)
activity_map = dict(zip(activity_labels[0], activity_labels[1]))
activity_map_reverse = {v: k for k, v in activity_map.items()}

print(f"训练集: {len(train_x)} 样本, 测试集: {len(test_x)} 样本")

# ============================================
# 2. 特征提取函数
# ============================================

def extract_6_features(x_data, y_data, z_data):
    """提取6个时域特征"""
    n = len(x_data)
    feat = np.zeros((n, 6))
    
    for i in range(n):
        sx = x_data.iloc[i].values
        sy = y_data.iloc[i].values
        sz = z_data.iloc[i].values
        
        # 1. PeakX: X轴峰峰值
        feat[i, 0] = np.max(sx) - np.min(sx)
        
        # 2. StdMean: 三轴平均标准差
        feat[i, 1] = np.mean([np.std(sx), np.std(sy), np.std(sz)])
        
        # 3. RMS: 合向量有效值
        rms_x = np.sqrt(np.mean(sx**2))
        rms_y = np.sqrt(np.mean(sy**2))
        rms_z = np.sqrt(np.mean(sz**2))
        feat[i, 2] = np.sqrt(rms_x**2 + rms_y**2 + rms_z**2)
        
        # 4. MeanX: X轴均值
        feat[i, 3] = np.mean(sx)
        
        # 5. SMA: 信号幅度区域
        feat[i, 4] = np.mean(np.abs(sx) + np.abs(sy) + np.abs(sz))
        
        # 6. ZCR_X: X轴过零率
        zc = np.where(np.diff(np.signbit(sx)))[0]
        feat[i, 5] = len(zc) / len(sx)
    
    return feat

print("\n🔧 2. 提取特征...")
X_train = extract_6_features(train_x, train_y, train_z)
X_test = extract_6_features(test_x, test_y, test_z)

y_train_labels = y_train[0].values
y_test_labels = y_test[0].values

feature_names = ['PeakX', 'StdMean', 'RMS', 'MeanX', 'SMA', 'ZCR_X']
print(f"特征矩阵: {X_train.shape}")

# ============================================
# 3. GMM分类器
# ============================================

class GMMClassifier:
    """
    高斯混合模型分类器
    为每个类别独立训练一个GMM，预测时选似然最大的类别
    """
    
    def __init__(self, n_components_dict=None, covariance_type='full'):
        """
        n_components_dict: 每个类别的高斯组件数，如 {1:3, 2:2, 3:2, 4:3, 5:2, 6:1}
        covariance_type: 'full', 'tied', 'diag', 'spherical'
        """
        self.n_components_dict = n_components_dict
        self.covariance_type = covariance_type
        self.gmms = {}
        self.classes = None
    
    def fit(self, X, y):
        """训练：为每个类别训练一个GMM"""
        self.classes = np.unique(y)
        
        for cls in self.classes:
            X_cls = X[y == cls]
            
            # 确定组件数
            if self.n_components_dict and cls in self.n_components_dict:
                n_comp = self.n_components_dict[cls]
            else:
                n_comp = 2  # 默认2个组件
            
            # 训练GMM
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=self.covariance_type,
                random_state=42,
                max_iter=200
            )
            gmm.fit(X_cls)
            self.gmms[cls] = gmm
            
            print(f"   类别 {cls} ({activity_map[cls]}): 组件数={n_comp}, 样本数={len(X_cls)}")
        
        return self
    
    def predict_proba(self, X):
        """返回每个样本属于每个类别的概率（归一化似然）"""
        n_samples = len(X)
        n_classes = len(self.classes)
        likelihoods = np.zeros((n_samples, n_classes))
        
        for i, cls in enumerate(self.classes):
            # 计算对数似然
            log_likelihood = self.gmms[cls].score_samples(X)
            likelihoods[:, i] = np.exp(log_likelihood)
        
        # 归一化得到概率
        proba = likelihoods / likelihoods.sum(axis=1, keepdims=True)
        return proba
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


# ============================================
# 4. 确定最优组件数
# ============================================

print("\n🔧 3. 确定每个类别的最优组件数...")

def find_optimal_components(X, y, max_components=5):
    """用BIC准则为每个类别选择最优组件数"""
    classes = np.unique(y)
    optimal = {}
    
    for cls in classes:
        X_cls = X[y == cls]
        bic_scores = []
        
        for n in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=n, random_state=42, max_iter=200)
            gmm.fit(X_cls)
            bic_scores.append(gmm.bic(X_cls))
        
        best_n = np.argmin(bic_scores) + 1
        optimal[cls] = best_n
        print(f"   {cls} ({activity_map[cls]}): 最优组件数 = {best_n}")
    
    return optimal

# 计算最优组件数（可选，用部分样本加速）
sample_size = min(2000, len(X_train))
indices = np.random.choice(len(X_train), sample_size, replace=False)
optimal_components = find_optimal_components(X_train[indices], y_train_labels[indices])
print(f"\n最终组件数配置: {optimal_components}")

# ============================================
# 5. 训练GMM分类器
# ============================================

print("\n🌳 4. 训练GMM分类器...")

# 可以用自动计算的，也可以手动指定
# optimal_components = {1:3, 2:2, 3:2, 4:2, 5:2, 6:2}  # 手动指定

clf = GMMClassifier(n_components_dict=optimal_components, covariance_type='full')
clf.fit(X_train, y_train_labels)

# ============================================
# 6. 测试集评估
# ============================================

print("\n📊 5. 测试集评估...")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test_labels, y_pred)

print(f"\n✅ 测试集准确率: {accuracy*100:.2f}%")

print("\n📋 分类报告:")
print(classification_report(y_test_labels, y_pred, 
                           target_names=[activity_map[i] for i in range(1, 7)]))

# ============================================
# 7. 混淆矩阵
# ============================================

print("\n📈 6. 混淆矩阵...")

cm = confusion_matrix(y_test_labels, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=[activity_map[i] for i in range(1, 7)],
            yticklabels=[activity_map[i] for i in range(1, 7)])
plt.xlabel('预测', fontsize=12)
plt.ylabel('真实', fontsize=12)
plt.title(f'GMM分类器混淆矩阵 (准确率: {accuracy*100:.1f}%)', fontsize=12)
plt.tight_layout()
plt.savefig('gmm_confusion_matrix.png', dpi=150)
plt.show()

# ============================================
# 8. 与决策树对比
# ============================================

print("\n🔄 7. 与决策树对比...")

from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_clf.fit(X_train, y_train_labels)
dt_acc = accuracy_score(y_test_labels, dt_clf.predict(X_test))

print(f"\n决策树准确率: {dt_acc*100:.2f}%")
print(f"GMM准确率:   {accuracy*100:.2f}%")
print(f"差值: {(accuracy - dt_acc)*100:.2f}%")

# ============================================
# 9. 可视化：GMM拟合效果（PCA降维到2D）
# ============================================

print("\n🎨 8. 可视化GMM拟合效果...")

# 选择两个类别进行可视化
vis_classes = [1, 4]  # 走路和坐着
vis_indices = np.where(np.isin(y_train_labels, vis_classes))[0]
X_vis = X_train[vis_indices]
y_vis = y_train_labels[vis_indices]

# PCA降维到2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_vis)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, cls in enumerate(vis_classes):
    ax = axes[idx]
    mask = y_vis == cls
    X_cls_pca = X_pca[mask]
    
    # 散点图
    ax.scatter(X_cls_pca[:, 0], X_cls_pca[:, 1], alpha=0.5, label=f'{activity_map[cls]}')
    
    # 训练该类的GMM并绘制等高线
    gmm_vis = GaussianMixture(n_components=optimal_components[cls], random_state=42)
    gmm_vis.fit(X_train[y_train_labels == cls])
    
    # 生成网格
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # 注意：需要在PCA空间绘制，这里简化处理
    ax.set_title(f'{activity_map[cls]} - GMM拟合 (组件数={optimal_components[cls]})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('GMM拟合效果可视化 (PCA降维)', fontsize=14)
plt.tight_layout()
plt.savefig('gmm_fit_visualization.png', dpi=150)
plt.show()

# ============================================
# 10. 总结
# ============================================

print("\n" + "="*70)
print("📋 实验结果总结")
print("="*70)

print(f"""
测试集准确率: {accuracy*100:.2f}%

与决策树对比:
   GMM分类器:     {accuracy*100:.2f}%
   决策树:        {dt_acc*100:.2f}%
   {'GMM优于决策树' if accuracy > dt_acc else '决策树优于GMM'}

各类别组件数配置:
   {optimal_components}

特点:
   - GMM假设每个类别的特征向量服从高斯混合分布
   - 为每个类别独立建模，能捕捉类别内部的子模式
   - 输出概率，可解释性强
""")