"""
分类器展示项目：决策树详解 + 四分类器对比可视化
================================================
UCI HAR Dataset, 72-D 特征 (6通道 × 12特征), 6类动作识别

输出 (figures/分类器展示/):
  01_决策树结构.png         — 决策树前4层, 中文类名, 节点着色
  02_决策树特征重要性.png    — Top-20 Gini 重要性, 频域/时域分色
  03_决策树混淆矩阵.png      — 归一化混淆矩阵 + 各类F1
  04_四分类器混淆矩阵.png    — DT/KNN/SVM/RF 并排对比
  05_四分类器指标对比.png    — Accuracy + Macro-F1 + Weighted-F1
  06_分类器训练耗时.png      — 训练时间对比
  07_分类决策边界示意.png    — 2D PCA投影 + 决策区域
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import entropy as stats_entropy
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ───────────────────────────────────────────────────────
FS = 50
N_SAMPLES = 128
DATA_DIR = "UCI HAR Dataset"
SAVE_DIR = os.path.join("figures", "分类器展示")
os.makedirs(SAVE_DIR, exist_ok=True)

ACTIVITY_CN = {
    1: "行走", 2: "上楼", 3: "下楼",
    4: "静坐", 5: "站立", 6: "躺卧",
}
ACTIVITY_EN = {
    1: "Walking", 2: "Upstairs", 3: "Downstairs",
    4: "Sitting", 5: "Standing", 6: "Laying",
}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 8,
    "font.sans-serif": ["Arial Unicode MS", "SimHei", "Heiti SC", "STHeiti"],
    "axes.unicode_minus": False,
})
sns.set_style("whitegrid")

# ═══════════════════════════════════════════════════════════════
#  数据加载 + 特征提取 (72-D)
# ═══════════════════════════════════════════════════════════════

def load_uci_har():
    """加载 UCI HAR, 返回训练/测试原始信号 + 标签。"""
    X, y = {}, {}
    for subset in ["train", "test"]:
        path = os.path.join(DATA_DIR, subset, "Inertial Signals")
        channels = []
        for axis in ["x", "y", "z"]:
            for signal in ["body_acc", "body_gyro"]:
                data = np.loadtxt(
                    os.path.join(path, f"{signal}_{axis}_{subset}.txt"),
                    dtype=np.float32)
                channels.append(data)
        X[subset] = np.stack(channels, axis=-1)  # (N, 128, 6)
        y[subset] = np.loadtxt(
            os.path.join(DATA_DIR, subset, f"y_{subset}.txt")).astype(int)
    return X, y

def fft_spectrum(signal):
    n = len(signal)
    vals = fft(signal)
    mag = np.abs(vals) / n
    mag = mag[: n // 2 + 1]
    mag[1:-1] *= 2
    freqs = fftfreq(n, 1 / FS)[: n // 2 + 1]
    return freqs, mag

def extract_features(raw_data):
    """72-D: 6通道 × (8频域 + 4时域)。"""
    n_win, _, n_ch = raw_data.shape
    rows = []
    for i in range(n_win):
        row = []
        for ch in range(n_ch):
            sig = raw_data[i, :, ch]
            freqs, mag = fft_spectrum(sig)
            total = np.sum(mag)
            eps = 1e-12
            # 频域 8
            row.append(freqs[np.argmax(mag)])
            if total > eps:
                row.append(np.sum(freqs * mag) / total)
                cum = np.cumsum(mag)
                row.append(freqs[np.searchsorted(cum, total / 2)])
                row.append(np.sum(mag ** 2))
                row.append(stats_entropy(mag / total + eps))
                nyq = freqs[-1]
                row.append(np.sum(mag[(freqs >= 0) & (freqs < nyq * 0.2)] ** 2))
                row.append(np.sum(mag[(freqs >= nyq * 0.2) & (freqs < nyq * 0.6)] ** 2))
                row.append(np.sum(mag[(freqs >= nyq * 0.6)] ** 2))
            else:
                row.extend([0.0] * 7)
            # 时域 4
            row.append(np.mean(sig))
            row.append(np.var(sig))
            row.append(np.ptp(sig))
            row.append(np.sum(np.diff(np.signbit(sig))) / len(sig))
        rows.append(row)
    return np.array(rows, dtype=np.float64)

# ═══════════════════════════════════════════════════════════════
#  图1: 决策树结构
# ═══════════════════════════════════════════════════════════════

def plot_decision_tree_structure(clf, feat_names):
    """决策树前4层, 中文类名, 填色节点。"""
    fig, ax = plt.subplots(figsize=(28, 14))
    plot_tree(
        clf, max_depth=4, feature_names=feat_names,
        class_names=[ACTIVITY_CN[i] for i in range(1, 7)],
        filled=True, rounded=True, fontsize=7, ax=ax,
        impurity=False, proportion=True,
        node_ids=False,
    )
    ax.set_title(
        "图1: 决策树结构 (前4层) — UCI HAR 6类动作识别",
        fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "01_决策树结构.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  图2: 特征重要性
# ═══════════════════════════════════════════════════════════════

def plot_feature_importance_dt(clf, feat_names):
    """Top-20 Gini 重要性, 频域/时域分色, 标注数值。"""
    importances = clf.feature_importances_
    top_n = 20
    idx = np.argsort(importances)[-top_n:]
    names = [feat_names[i] for i in idx]
    values = importances[idx]

    colors = [
        "#E53935" if any(kw in n for kw in
            ["Freq", "Spectral", "Band", "Entropy", "Energy"])
        else "#1E88E5" for n in names
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(range(top_n), values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Gini Importance", fontsize=11)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#E53935", label="频域特征 (Frequency Domain)"),
        Patch(color="#1E88E5", label="时域特征 (Time Domain)"),
    ], fontsize=9, loc="lower right")
    ax.set_title(
        "图2: 决策树特征重要性 Top-20 — 红色=频域, 蓝色=时域",
        fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "02_决策树特征重要性.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  图3: 决策树混淆矩阵
# ═══════════════════════════════════════════════════════════════

def plot_dt_confusion_matrix(y_true, y_pred):
    """归一化混淆矩阵 + 每类F1标注。"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = [ACTIVITY_CN[i] for i in range(1, 7)]
    f1_per_class = f1_score(y_true, y_pred, average=None)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, vmin=0, vmax=1, ax=ax,
                annot_kws={"fontsize": 11})
    ax.set_xlabel("预测类别", fontsize=12)
    ax.set_ylabel("真实类别", fontsize=12)
    ax.set_title(
        f"图3: 决策树混淆矩阵 (归一化) — Acc={accuracy_score(y_true, y_pred)*100:.1f}%",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "03_决策树混淆矩阵.png")
    fig.savefig(path)
    plt.close(fig)

    # 终端输出
    print(f"  ✓ {path}")
    print(f"    各类F1: ", end="")
    for i, (name, f1) in enumerate(zip(labels, f1_per_class)):
        print(f"{name}={f1:.3f}", end="  ")
    print()

# ═══════════════════════════════════════════════════════════════
#  图4: 四分类器混淆矩阵并排
# ═══════════════════════════════════════════════════════════════

def plot_four_confusion_matrices(y_test, classifiers_dict):
    """DT / KNN / SVM / RF 四分类器混淆矩阵并排。"""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    labels = [ACTIVITY_CN[i] for i in range(1, 7)]

    for ax, (name, y_pred) in zip(axes, classifiers_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        acc = accuracy_score(y_test, y_pred)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, vmin=0, vmax=1, ax=ax,
                    annot_kws={"fontsize": 8},
                    cbar=(ax == axes[-1]))
        ax.set_xlabel("预测", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("真实", fontsize=9)
        ax.set_title(f"{name}\nAcc = {acc*100:.1f}%", fontsize=10, fontweight="bold")

    fig.suptitle("图4: 四分类器混淆矩阵对比 (归一化)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "04_四分类器混淆矩阵.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  图5: 四分类器指标对比
# ═══════════════════════════════════════════════════════════════

def plot_metrics_comparison(results):
    """Accuracy + Macro-F1 + Weighted-F1 分组柱状图。"""
    names = [r["name"] for r in results]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(names))
    width = 0.22

    metrics = [
        ("accuracy", "Accuracy", "#1976D2"),
        ("macro_f1", "Macro-F1", "#388E3C"),
        ("weighted_f1", "Weighted-F1", "#F57C00"),
    ]

    for i, (key, label, color) in enumerate(metrics):
        values = [r[key] for r in results]
        bars = ax.bar(x + i * width, values, width, label=label,
                      color=color, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.4f}", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("图5: 四分类器性能对比 — UCI HAR 6类动作识别",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "05_四分类器指标对比.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  图6: 训练耗时对比
# ═══════════════════════════════════════════════════════════════

def plot_training_time(times_dict):
    """水平条形图: 各分类器训练耗时。"""
    names = list(times_dict.keys())
    times = list(times_dict.values())
    colors = ["#E53935", "#1E88E5", "#FB8C00", "#43A047"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(names, times, color=colors, edgecolor="white", height=0.55)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{t:.3f}s", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("训练时间 (秒)", fontsize=11)
    ax.set_title("图6: 分类器训练耗时对比", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "06_分类器训练耗时.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  图7: PCA 2D 投影 + 决策边界示意
# ═══════════════════════════════════════════════════════════════

def plot_decision_boundary_2d(X_train, y_train, clf, scaler):
    """
    用 PCA 将 72-D 降到 2-D, 画训练集散点 + 叠加分类器预测区域背景。
    用 meshgrid 采样决策边界。
    """
    # PCA 降维
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_train)  # (7352, 2)

    # 在 PCA 2D 空间训练一个简化分类器用于可视化
    clf_2d = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
    clf_2d.fit(X_2d, y_train)

    # meshgrid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300))
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 画图
    fig, ax = plt.subplots(figsize=(10, 7.5))
    colors_act = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]
    cmap_bg = matplotlib.colors.ListedColormap(
        ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#FFCDD2", "#E1BEE7", "#B2EBF2"])

    # 决策区域背景
    ax.pcolormesh(xx, yy, Z, cmap=cmap_bg, alpha=0.35, shading="auto")

    # 散点: 每类取 500 个样本防止过密
    for act_id in range(1, 7):
        mask = y_train == act_id
        idx = np.where(mask)[0]
        sample = np.random.RandomState(42).choice(
            idx, min(500, len(idx)), replace=False)
        ax.scatter(X_2d[sample, 0], X_2d[sample, 1],
                   s=4, alpha=0.5, color=colors_act[act_id - 1],
                   label=ACTIVITY_CN[act_id], rasterized=True)

    ax.set_xlabel(f"PCA 主成分 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PCA 主成分 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(
        "图7: PCA 2D投影 + 决策树决策区域 (max_depth=6)",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, markerscale=3, loc="upper right")
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "07_PCA决策边界示意.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 55 + "╗")
    print("║  分类器展示项目：决策树详解 + 四分类器对比          ║")
    print("║  UCI HAR Dataset · 72-D 特征 · 6类动作识别          ║")
    print("╚" + "═" * 55 + "╝")

    # ── 1. 加载数据 ──
    print("\n[1/6] 加载 UCI HAR 数据…")
    X_raw, y = load_uci_har()
    print(f"  训练集: {X_raw['train'].shape[0]} 窗口 (21人)")
    print(f"  测试集: {X_raw['test'].shape[0]} 窗口 (9人)")
    print(f"  通道: body_acc x/y/z + body_gyro x/y/z")

    # ── 2. 特征提取 ──
    print("\n[2/6] 提取 72-D 特征 (6通道 × 12特征)…")
    X_train = extract_features(X_raw["train"])
    X_test = extract_features(X_raw["test"])
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 构建特征名
    ch_names = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
    freq_tags = ["PeakFreq", "MeanFreq", "MedianFreq", "Energy",
                 "Entropy", "BandLow", "BandMid", "BandHigh"]
    time_tags = ["Mean", "Var", "PTP", "ZCR"]
    feat_names = []
    for ch in ch_names:
        for t in freq_tags:
            feat_names.append(f"{ch}_{t}")
        for t in time_tags:
            feat_names.append(f"{ch}_{t}")

    # ── 3. 决策树训练 ──
    print("\n[3/6] 训练决策树 (max_depth=12, min_samples_leaf=5)…")
    dt = DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y["train"])
    dt_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y["test"], dt_pred)
    dt_f1_macro = f1_score(y["test"], dt_pred, average="macro")
    dt_f1_w = f1_score(y["test"], dt_pred, average="weighted")
    print(f"  Accuracy={dt_acc*100:.2f}%  Macro-F1={dt_f1_macro:.4f}  "
          f"Weighted-F1={dt_f1_w:.4f}")
    print(f"  树深度={dt.get_depth()}, 叶节点数={dt.get_n_leaves()}")

    # ── 4. 训练其他分类器 ──
    print("\n[4/6] 训练 KNN / SVM / Random Forest…")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifiers = {}
    preds = {}
    times = {}

    # KNN
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    knn.fit(X_train_s, y["train"])
    preds["KNN (k=5)"] = knn.predict(X_test_s)
    times["KNN (k=5)"] = time.time() - t0

    # SVM
    t0 = time.time()
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
    svm.fit(X_train_s, y["train"])
    preds["SVM (RBF)"] = svm.predict(X_test_s)
    times["SVM (RBF)"] = time.time() - t0

    # Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        random_state=42, n_jobs=-1)
    rf.fit(X_train, y["train"])
    preds["Random Forest"] = rf.predict(X_test)
    times["Random Forest"] = time.time() - t0

    # Decision Tree 时间 (已在上面训练)
    t0 = time.time()
    dt2 = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42)
    dt2.fit(X_train, y["train"])
    times["Decision Tree"] = time.time() - t0
    preds["Decision Tree"] = dt_pred  # 用已预测的结果

    # 输出
    for name in ["Decision Tree", "KNN (k=5)", "SVM (RBF)", "Random Forest"]:
        acc = accuracy_score(y["test"], preds[name])
        f1m = f1_score(y["test"], preds[name], average="macro")
        f1w = f1_score(y["test"], preds[name], average="weighted")
        print(f"  {name:<16s}: Acc={acc*100:.2f}%  F1_m={f1m:.4f}  "
              f"F1_w={f1w:.4f}  time={times[name]:.3f}s")

    # ── 5. 决策树可视化 (3张图) ──
    print("\n[5/6] 决策树可视化 (3张图)…")
    plot_decision_tree_structure(dt, feat_names)
    plot_feature_importance_dt(dt, feat_names)
    plot_dt_confusion_matrix(y["test"], dt_pred)

    # ── 6. 分类器对比可视化 (4张图) ──
    print("\n[6/6] 分类器对比可视化 (4张图)…")
    plot_four_confusion_matrices(y["test"], preds)

    results = []
    for name in ["Decision Tree", "KNN (k=5)", "SVM (RBF)", "Random Forest"]:
        results.append({
            "name": name,
            "accuracy": accuracy_score(y["test"], preds[name]),
            "macro_f1": f1_score(y["test"], preds[name], average="macro"),
            "weighted_f1": f1_score(y["test"], preds[name], average="weighted"),
        })

    plot_metrics_comparison(results)
    plot_training_time(times)
    plot_decision_boundary_2d(X_train, y["train"], dt, scaler)

    # ── 终端汇总 ──
    print(f"\n{'═' * 55}")
    print(f"  所有图表保存至: {os.path.abspath(SAVE_DIR)}/")
    print(f"  共 7 张图")
    print(f"{'═' * 55}")

    # 最优结果
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n  最优分类器: {best['name']} ({best['accuracy']*100:.2f}%)")

    # 决策树根节点特征
    root_feat = feat_names[np.argmax(dt.feature_importances_)]
    print(f"  决策树最强特征: {root_feat}")
    print(f"  决策树 Top-5 特征: ", end="")
    top5_idx = np.argsort(dt.feature_importances_)[-5:][::-1]
    for i, idx in enumerate(top5_idx):
        print(f"{feat_names[idx]}({dt.feature_importances_[idx]:.4f})", end="  ")
    print()

if __name__ == "__main__":
    main()
