import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import os

# 设置路径
data_path = "./UCI HAR Dataset/UCI HAR Dataset"

print("加载数据...")

# 加载训练集
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

print("提取特征...")

# 提取6个特征
def get_features(x_data, y_data, z_data):
    n = len(x_data)
    feat = np.zeros((n, 6))
    for i in range(n):
        sx = x_data.iloc[i].values
        sy = y_data.iloc[i].values
        sz = z_data.iloc[i].values
        
        feat[i, 0] = np.max(sx) - np.min(sx)
        feat[i, 1] = np.mean([np.std(sx), np.std(sy), np.std(sz)])
        rms_x = np.sqrt(np.mean(sx**2))
        rms_y = np.sqrt(np.mean(sy**2))
        rms_z = np.sqrt(np.mean(sz**2))
        feat[i, 2] = np.sqrt(rms_x**2 + rms_y**2 + rms_z**2)
        feat[i, 3] = np.mean(sx)
        feat[i, 4] = np.mean(np.abs(sx) + np.abs(sy) + np.abs(sz))
        zc = np.where(np.diff(np.signbit(sx)))[0]
        feat[i, 5] = len(zc) / len(sx)
    return feat

X_train = get_features(train_x, train_y, train_z)
X_test = get_features(test_x, test_y, test_z)

y_train_labels = y_train[0].values
y_test_labels = y_test[0].values

print("训练决策树...")

clf = DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=42)
clf.fit(X_train, y_train_labels)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test_labels, y_pred)
print(f"准确率: {acc*100:.1f}%")

# 画图 - 大图 + 小字 + 拉开间距
feature_names = ['PeakX', 'StdMean', 'RMS', 'MeanX', 'SMA', 'ZCR_X']
class_names = ['WALK', 'UP', 'DOWN', 'SIT', 'STAND', 'LAY']

plt.figure(figsize=(30, 18))  # 图更大
plot_tree(clf, 
          feature_names=feature_names, 
          class_names=class_names, 
          filled=True, 
          rounded=True, 
          fontsize=8,           # 字体更小
          impurity=False,
          proportion=False,
          precision=2)
plt.title(f'Decision Tree (Accuracy: {acc*100:.1f}%)', fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')  # 更高分辨率
plt.show()

print("图片已保存: decision_tree.png")

# 再加一个文本版，保证能看清规则
print("\n" + "="*70)
print("决策树规则（文本版）")
print("="*70)

activity_names = ['走路', '上楼', '下楼', '坐着', '站着', '躺着']

def print_tree(node, indent=""):
    if clf.tree_.children_left[node] == -1:  # 叶子
        val = clf.tree_.value[node][0]
        pred = np.argmax(val)
        print(f"{indent}→ {activity_names[pred]}")
    else:
        feat = feature_names[clf.tree_.feature[node]]
        thresh = clf.tree_.threshold[node]
        print(f"{indent}如果 {feat} <= {thresh:.3f}:")
        print_tree(clf.tree_.children_left[node], indent + "    ")
        print(f"{indent}否则 ({feat} > {thresh:.3f}):")
        print_tree(clf.tree_.children_right[node], indent + "    ")

print_tree(0)