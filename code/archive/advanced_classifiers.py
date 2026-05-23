"""
进阶分类器对比：KNN / SVM / Decision Tree
=========================================
UCI HAR Dataset: 21 人训练 (7352 窗口) / 9 人测试 (2947 窗口)
特征: 6通道 × (8频域 + 4时域) = 72维
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import entropy as stats_entropy
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ─────────────────────────────────────────────────────
FS = 50
N_SAMPLES = 128
DATA_DIR = "UCI HAR Dataset"
SAVE_DIR = "figures/uci"
os.makedirs(SAVE_DIR, exist_ok=True)

ACTIVITY = {
    1: "Walking", 2: "Walking\nUpstairs", 3: "Walking\nDownstairs",
    4: "Sitting", 5: "Standing", 6: "Laying",
}
ACTIVITY_SHORT = {
    1: "Walk", 2: "Up", 3: "Down", 4: "Sit", 5: "Stand", 6: "Lay",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "legend.fontsize": 8,
})
sns.set_style("whitegrid")


# ═══════════════════════════════════════════════════════════════
#  数据加载 + 特征提取
# ═══════════════════════════════════════════════════════════════

def load_data():
    """加载 UCI HAR — 21人训练, 9人测试。"""
    X, y = {}, {}
    for subset in ["train", "test"]:
        path = os.path.join(DATA_DIR, subset, "Inertial Signals")
        channels = []
        for axis in ["x", "y", "z"]:
            for signal in ["body_acc", "body_gyro"]:
                data = np.loadtxt(os.path.join(path, f"{signal}_{axis}_{subset}.txt"), dtype=np.float32)
                channels.append(data)
        X[subset] = np.stack(channels, axis=-1)
        y[subset] = np.loadtxt(os.path.join(DATA_DIR, subset, f"y_{subset}.txt")).astype(int)
    return X, y


def fft_spectrum(signal):
    n = len(signal)
    vals = fft(signal)
    mag = np.abs(vals) / n
    mag = mag[: n // 2 + 1]
    mag[1:-1] *= 2
    freqs = fftfreq(n, 1 / FS)[: n // 2 + 1]
    return freqs, mag


def extract_features(raw_data, verbose=True):
    """逐窗口提取 72 维特征 (同 main.py 流水线)。"""
    n_windows = raw_data.shape[0]
    n_channels = raw_data.shape[2]
    all_rows = []

    for i in range(n_windows):
        row = []
        for ch in range(n_channels):
            signal = raw_data[i, :, ch]
            freqs, mag = fft_spectrum(signal)
            # 8 频域特征
            total = np.sum(mag)
            eps = 1e-12
            row.append(freqs[np.argmax(mag)])  # PeakFreq
            if total > eps:
                row.append(np.sum(freqs * mag) / total)  # MeanFreq
                cum = np.cumsum(mag)
                row.append(freqs[np.searchsorted(cum, total / 2)])  # MedianFreq
                row.append(np.sum(mag ** 2))  # SpectralEnergy
                row.append(stats_entropy(mag / total + eps))  # SpectralEntropy
                nyq = freqs[-1]
                row.append(np.sum(mag[(freqs >= 0) & (freqs < nyq * 0.2)] ** 2))  # BandLow
                row.append(np.sum(mag[(freqs >= nyq * 0.2) & (freqs < nyq * 0.6)] ** 2))  # BandMid
                row.append(np.sum(mag[(freqs >= nyq * 0.6)] ** 2))  # BandHigh
            else:
                row.extend([0.0] * 7)
            # 4 时域特征
            row.append(np.mean(signal))
            row.append(np.var(signal))
            row.append(np.ptp(signal))
            row.append(np.sum(np.diff(np.signbit(signal))) / len(signal))
        all_rows.append(row)

        if verbose and (i + 1) % 2000 == 0:
            print(f"  特征提取: {i + 1}/{n_windows}")

    return np.array(all_rows, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrices(y_test, preds, save_path):
    """三分类器混淆矩阵并排。"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    titles = ["Decision Tree", "KNN (k=5)", "SVM (RBF)"]
    names_short = [ACTIVITY_SHORT[i] for i in range(1, 7)]

    for ax, (name, y_pred) in zip(axes, zip(titles, preds)):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=names_short, yticklabels=names_short,
                    linewidths=0.5, annot_kws={"fontsize": 9},
                    vmin=0, vmax=cm.max())
        ax.set_xlabel("Predicted")
        if ax == axes[0]:
            ax.set_ylabel("True")
        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f"{name}\nAccuracy = {acc * 100:.1f}%", fontsize=11)

    fig.suptitle("Confusion Matrices: Decision Tree vs KNN vs SVM", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


def plot_metrics_bar(results, save_path):
    """三分类器指标柱状图: Accuracy / Macro-F1 / Weighted-F1。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results))
    width = 0.2
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    labels = ["Accuracy", "Macro F1", "Weighted F1"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [r[metric] for r in results]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([r["name"] for r in results], fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Classifier Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


def plot_knn_k_selection(X_train, y_train, X_test, y_test, save_path):
    """KNN 不同 k 值的准确率曲线。"""
    ks = [1, 3, 5, 7, 9, 11, 15, 20, 30, 50]
    train_scores, test_scores = [], []
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, knn.predict(X_train)))
        test_scores.append(accuracy_score(y_test, knn.predict(X_test)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, train_scores, "o-", color="#2196F3", linewidth=1.5, label="Train")
    ax.plot(ks, test_scores, "s-", color="#FF5722", linewidth=1.5, label="Test")
    ax.axvline(5, color="gray", linestyle=":", alpha=0.6)
    ax.text(5.5, test_scores[ks.index(5)], f"k=5: {test_scores[ks.index(5)] * 100:.1f}%",
            fontsize=9, color="gray")
    ax.set_xlabel("k (neighbors)")
    ax.set_ylabel("Accuracy")
    ax.set_title("KNN: Accuracy vs k (weights=distance)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  进阶分类器对比: KNN / SVM / Decision Tree")
    print("  UCI HAR: 21人训练 (7352窗) / 9人测试 (2947窗)")
    print("═" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载 UCI HAR 原始惯性信号 (6通道)…")
    X_raw, y = load_data()
    print(f"  训练: {X_raw['train'].shape[0]} 窗口, 测试: {X_raw['test'].shape[0]} 窗口")
    for sub in ["train", "test"]:
        from collections import Counter
        cnt = Counter(y[sub])
        print(f"  {sub}: {dict(sorted(cnt.items()))}")

    # 2. 特征提取
    print("\n[2/5] 特征提取 (72维: 6通道 × 12特征)…")
    X_train = extract_features(X_raw["train"])
    X_test = extract_features(X_raw["test"])
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 3. 标准化 (SVM 和 KNN 需要)
    print("\n[3/5] 标准化 + 训练三分类器…")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Decision Tree — 无需标准化
    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y["train"])
    dt_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y["test"], dt_pred)

    # KNN — k=5, distance-weighted
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    knn.fit(X_train_scaled, y["train"])
    knn_pred = knn.predict(X_test_scaled)
    knn_acc = accuracy_score(y["test"], knn_pred)

    # SVM — RBF kernel, C=10
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y["train"])
    svm_pred = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y["test"], svm_pred)

    # 4. 评估
    print("\n[4/5] 分类结果…")
    results = []
    for name, y_pred, clf in [
        ("Decision Tree", dt_pred, dt),
        ("KNN (k=5)", knn_pred, knn),
        ("SVM (RBF)", svm_pred, svm),
    ]:
        acc = accuracy_score(y["test"], y_pred)
        macro_f1 = f1_score(y["test"], y_pred, average="macro")
        w_f1 = f1_score(y["test"], y_pred, average="weighted")
        results.append({"name": name, "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": w_f1})
        print(f"\n  ── {name} ──")
        print(f"  Accuracy={acc * 100:.2f}%  Macro-F1={macro_f1:.4f}  Weighted-F1={w_f1:.4f}")

    # 分类报告
    for name, y_pred in [("Decision Tree", dt_pred), ("KNN (k=5)", knn_pred), ("SVM (RBF)", svm_pred)]:
        print(f"\n  ── {name} 分类报告 ──")
        print(classification_report(
            y["test"], y_pred,
            target_names=[ACTIVITY[i].replace("\n", " ") for i in range(1, 7)],
            digits=4,
        ))

    # 5. 可视化
    print("\n[5/5] 生成对比图…")
    preds = [dt_pred, knn_pred, svm_pred]
    plot_confusion_matrices(y["test"], preds, os.path.join(SAVE_DIR, "advanced_cm_comparison.png"))
    plot_metrics_bar(results, os.path.join(SAVE_DIR, "advanced_metrics_bar.png"))
    plot_knn_k_selection(X_train_scaled, y["train"], X_test_scaled, y["test"],
                         os.path.join(SAVE_DIR, "advanced_knn_k_selection.png"))

    print("\n" + "═" * 60)
    print("  完成。图像保存至 figures/uci/")
    print("═" * 60)


if __name__ == "__main__":
    main()
