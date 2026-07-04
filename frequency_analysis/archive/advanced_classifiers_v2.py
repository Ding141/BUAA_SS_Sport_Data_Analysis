"""
进阶分类器 V2: 增强特征集 + KNN / SVM / Decision Tree / Random Forest
=====================================================================
新增特征:
  频域 (8→20维/通道):
    - 原有: PeakFreq, MeanFreq, MedianFreq, Energy, Entropy, 3×Band
    - 新增: PeakMag, Spread, Skewness, Kurtosis, Flatness, Crest,
            Rolloff75/90/95, 7×FineBand(0-1,1-3,3-5,5-8,8-12,12-18,18-25Hz)
  时域 (4→11维/通道):
    - 原有: Mean, Var, PTP, ZCR
    - 新增: RMS, Skewness, Kurtosis, IQR, Median, Max, Min

总计: 6通道 × 31特征 = 186维 (vs 原72维)
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import entropy as stats_entropy, skew, kurtosis, iqr
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
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
#  数据加载
# ═══════════════════════════════════════════════════════════════

def load_data():
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


# ═══════════════════════════════════════════════════════════════
#  增强特征提取 (186维)
# ═══════════════════════════════════════════════════════════════

FREQ_BANDS = [
    (0, 1), (1, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 25),
]
BAND_LABELS = ["B_0-1Hz", "B_1-3Hz", "B_3-5Hz", "B_5-8Hz",
               "B_8-12Hz", "B_12-18Hz", "B_18-25Hz"]

FEATURE_NAMES_CH = []
for domain in ["Freq", "Time"]:
    names = {
        "Freq": [
            "PeakFreq", "MeanFreq", "MedianFreq", "Energy", "Entropy",
            "BandLow", "BandMid", "BandHigh",
            "PeakMag", "Spread", "Skewness", "Kurtosis",
            "Flatness", "Crest", "Rolloff75", "Rolloff90", "Rolloff95",
        ] + BAND_LABELS,
        "Time": [
            "Mean", "Var", "PTP", "ZCR",
            "RMS", "Skewness", "Kurtosis", "IQR", "Median", "Max", "Min",
        ],
    }
    FEATURE_NAMES_CH.extend(names[domain])

CHANNEL_NAMES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
FEATURE_NAMES = []
for ch in CHANNEL_NAMES:
    for fn in FEATURE_NAMES_CH:
        FEATURE_NAMES.append(f"{ch}_{fn}")


def extract_freq_features(freqs, mag, eps=1e-12):
    """提取单个通道的 24 个频域特征。"""
    total = np.sum(mag)
    feats = []

    # ── 基础 (5) ──
    feats.append(freqs[np.argmax(mag)])                      # PeakFreq
    if total > eps:
        feats.append(np.sum(freqs * mag) / total)            # MeanFreq
        cum = np.cumsum(mag)
        feats.append(freqs[np.searchsorted(cum, total / 2)]) # MedianFreq
    else:
        feats.extend([0.0, 0.0])

    feats.append(np.sum(mag ** 2))                           # SpectralEnergy
    if total > eps:
        feats.append(stats_entropy(mag / total + eps))       # SpectralEntropy
    else:
        feats.append(0.0)

    # ── 原有3频带 (3) ──
    nyq = freqs[-1]
    b_low  = np.sum(mag[(freqs >= 0) & (freqs < nyq * 0.2)] ** 2)
    b_mid  = np.sum(mag[(freqs >= nyq * 0.2) & (freqs < nyq * 0.6)] ** 2)
    b_high = np.sum(mag[(freqs >= nyq * 0.6)] ** 2)
    feats.extend([b_low, b_mid, b_high])

    # ── 新增: 谱形状 (7) ──
    feats.append(np.max(mag))                                # PeakMag
    if total > eps:
        centroid = feats[1]  # MeanFreq
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / total)
        feats.append(spread)                                 # SpectralSpread
        feats.append(np.sum(((freqs - centroid) ** 3) * mag) / (total * (spread + eps) ** 3))  # Skewness
        feats.append(np.sum(((freqs - centroid) ** 4) * mag) / (total * (spread + eps) ** 4))  # Kurtosis
        feats.append(np.exp(np.mean(np.log(mag + eps))) / (np.mean(mag) + eps))  # Flatness
        feats.append(np.max(mag) / (np.mean(mag) + eps))     # Crest
        # Rolloff points
        cum_norm = cum / total
        feats.append(freqs[np.searchsorted(cum_norm, 0.75)]) # Rolloff75
        feats.append(freqs[np.searchsorted(cum_norm, 0.90)]) # Rolloff90
        feats.append(freqs[np.searchsorted(cum_norm, 0.95)]) # Rolloff95
    else:
        feats.extend([0.0] * 7)

    # ── 新增: 7细粒度频带 (7) ──
    for flo, fhi in FREQ_BANDS:
        mask = (freqs >= flo) & (freqs < fhi)
        feats.append(np.sum(mag[mask] ** 2))

    return feats  # 24 features


def extract_time_features(signal):
    """提取单个通道的 11 个时域特征。"""
    feats = []
    # 原有 (4)
    feats.append(np.mean(signal))
    feats.append(np.var(signal))
    feats.append(np.ptp(signal))
    feats.append(np.sum(np.diff(np.signbit(signal))) / len(signal))
    # 新增 (7)
    feats.append(np.sqrt(np.mean(signal ** 2)))              # RMS
    feats.append(skew(signal))                               # Skewness
    feats.append(kurtosis(signal))                           # Kurtosis
    feats.append(iqr(signal))                                # IQR
    feats.append(np.median(signal))                          # Median
    feats.append(np.max(signal))                             # Max
    feats.append(np.min(signal))                             # Min
    return feats


def extract_features_enhanced(raw_data, verbose=True):
    """逐窗口提取 186 维特征 (6通道 × 31特征)。"""
    n_windows = raw_data.shape[0]
    n_channels = raw_data.shape[2]
    all_rows = []

    for i in range(n_windows):
        row = []
        for ch in range(n_channels):
            signal = raw_data[i, :, ch]
            freqs, mag = fft_spectrum(signal)
            row.extend(extract_freq_features(freqs, mag))
            row.extend(extract_time_features(signal))
        all_rows.append(row)

        if verbose and (i + 1) % 2000 == 0:
            print(f"  特征提取: {i + 1}/{n_windows}")

    return np.array(all_rows, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrices(y_test, preds, names, save_path):
    n = len(preds)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 5.5))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()
    names_short = [ACTIVITY_SHORT[i] for i in range(1, 7)]

    for ax, (name, y_pred) in zip(axes, zip(names, preds)):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=names_short, yticklabels=names_short,
                    linewidths=0.5, annot_kws={"fontsize": 9},
                    vmin=0, vmax=cm.max())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f"{name}\nAccuracy = {acc * 100:.1f}%", fontsize=11)

    for ax in axes[len(preds):]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices — Enhanced Features (186-D)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


def plot_metrics_bar(results, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
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
                    f"{val:.4f}", ha="center", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([r["name"] for r in results], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Classifier Performance — Enhanced Features (186-D)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


def plot_feature_importance(rf, top_n=30, save_path=None):
    """Random Forest 特征重要性 Top-N。"""
    importances = rf.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    names_sub = [FEATURE_NAMES[i] for i in idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(top_n), importances[idx], color="#2196F3", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names_sub, fontsize=7)
    ax.set_xlabel("Gini Importance")
    ax.set_title(f"Random Forest — Top {top_n} Feature Importances", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  ✓ {save_path}")


def plot_accuracy_vs_v1(results_v1, results_v2, save_path):
    """V1 vs V2 准确率对比。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["name"] for r in results_v2]
    v1_acc = {r["name"]: r["accuracy"] for r in results_v1}
    v2_acc = [r["accuracy"] for r in results_v2]

    x = np.arange(len(names))
    width = 0.3

    bars1 = ax.bar(x - width / 2, [v1_acc.get(n, 0) for n in names], width,
                   color="#90CAF9", edgecolor="white", label="V1 (72-D)")
    bars2 = ax.bar(x + width / 2, v2_acc, width,
                   color="#1976D2", edgecolor="white", label="V2 (186-D)")

    for bar, val in zip(bars1, [v1_acc.get(n, 0) for n in names]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, v2_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9)
        idx = list(v2_acc).index(val)
        gain = v2_acc[idx] - v1_acc.get(names[idx], 0)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.04,
                f"+{gain:.3f}", ha="center", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy Improvement: V1 (72-D) vs V2 (186-D)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  进阶分类器 V2: 增强特征集 (186-D)")
    print("  新增: PeakMag / Spread / Skew / Kurt / Flatness / Crest")
    print("        Rolloff75/90/95 / 7细粒度频带 / RMS / IQR / Max/Min…")
    print("═" * 60)

    # 1. 加载
    print("\n[1/5] 加载 UCI HAR 原始惯性信号 (6通道)…")
    X_raw, y = load_data()
    print(f"  训练: {X_raw['train'].shape[0]} 窗口, 测试: {X_raw['test'].shape[0]} 窗口")

    # 2. 特征提取 (186-D)
    print(f"\n[2/5] 增强特征提取 (目标: {len(FEATURE_NAMES)}维)…")
    X_train = extract_features_enhanced(X_raw["train"])
    X_test = extract_features_enhanced(X_raw["test"])
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 3. 标准化 + 训练
    print("\n[3/5] 标准化 + 训练四分类器…")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y["train"])
    dt_pred = dt.predict(X_test)

    # Random Forest (新增)
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y["train"])
    rf_pred = rf.predict(X_test)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    knn.fit(X_train_scaled, y["train"])
    knn_pred = knn.predict(X_test_scaled)

    # SVM
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y["train"])
    svm_pred = svm.predict(X_test_scaled)

    # 4. 评估
    print("\n[4/5] 分类结果…")
    results = []
    for name, y_pred in [
        ("Decision Tree", dt_pred),
        ("Random Forest", rf_pred),
        ("KNN (k=5)", knn_pred),
        ("SVM (RBF)", svm_pred),
    ]:
        acc = accuracy_score(y["test"], y_pred)
        macro_f1 = f1_score(y["test"], y_pred, average="macro")
        w_f1 = f1_score(y["test"], y_pred, average="weighted")
        results.append({"name": name, "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": w_f1})
        print(f"  ── {name} ──")
        print(f"  Accuracy={acc * 100:.2f}%  Macro-F1={macro_f1:.4f}  Weighted-F1={w_f1:.4f}")

    # 详细分类报告
    for name, y_pred in [
        ("Decision Tree", dt_pred),
        ("Random Forest", rf_pred),
        ("KNN (k=5)", knn_pred),
        ("SVM (RBF)", svm_pred),
    ]:
        print(f"\n  ── {name} 分类报告 ──")
        print(classification_report(
            y["test"], y_pred,
            target_names=[ACTIVITY[i].replace("\n", " ") for i in range(1, 7)],
            digits=4,
        ))

    # 5. 可视化
    print("\n[5/5] 生成对比图…")
    preds = [dt_pred, rf_pred, knn_pred, svm_pred]
    clf_names = ["Decision Tree", "Random Forest", "KNN (k=5)", "SVM (RBF)"]
    plot_confusion_matrices(y["test"], preds, clf_names,
                            os.path.join(SAVE_DIR, "v2_cm_comparison.png"))
    plot_metrics_bar(results,
                     os.path.join(SAVE_DIR, "v2_metrics_bar.png"))
    plot_feature_importance(rf, top_n=30,
                            save_path=os.path.join(SAVE_DIR, "v2_feature_importance.png"))

    # V1 vs V2 对比
    results_v1 = [
        {"name": "Decision Tree", "accuracy": 0.7767, "macro_f1": 0.7730, "weighted_f1": 0.7749},
        {"name": "KNN (k=5)", "accuracy": 0.7913, "macro_f1": 0.7912, "weighted_f1": 0.7910},
        {"name": "SVM (RBF)", "accuracy": 0.8561, "macro_f1": 0.8578, "weighted_f1": 0.8565},
    ]
    results_v2_mapped = [r for r in results if r["name"] != "Random Forest"]
    plot_accuracy_vs_v1(results_v1, results_v2_mapped,
                        os.path.join(SAVE_DIR, "v2_v1_comparison.png"))

    # 终端对比表
    print("\n" + "═" * 70)
    print("  V1 (72-D) vs V2 (186-D) 准确率对比")
    print("═" * 70)
    print(f"  {'分类器':<20s} {'V1 (72-D)':<12s} {'V2 (186-D)':<12s} {'提升':<10s}")
    print("  " + "─" * 54)
    for rv2 in results:
        name = rv2["name"]
        v1_acc = results_v1[0]["accuracy"]  # fallback
        for rv1 in results_v1:
            if rv1["name"] == name:
                v1_acc = rv1["accuracy"]
                break
        if name not in [r["name"] for r in results_v1]:
            v1_acc = None
        if v1_acc is not None:
            gain = rv2["accuracy"] - v1_acc
            print(f"  {name:<20s} {v1_acc*100:>7.2f}%     {rv2['accuracy']*100:>7.2f}%     +{gain*100:.2f}pp")
        else:
            print(f"  {name:<20s} {'—':>11s}  {rv2['accuracy']*100:>7.2f}%     (new)")

    print("\n" + "═" * 60)
    print("  完成。图像保存至 figures/uci/")
    print("═" * 60)


if __name__ == "__main__":
    main()
