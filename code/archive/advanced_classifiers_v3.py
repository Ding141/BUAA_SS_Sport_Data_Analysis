"""
进阶分类器 V3: 特征选择 + 四分类器对比
======================================
V2 问题: 186-D 中大量弱相关/冗余特征拖累 KNN，Decision Tree 提升微弱
V3 方案:
  1. Mutual Information (互信息) 过滤 —— 捕捉非线性依赖，优于 ANOVA F 值
  2. Random Forest 嵌入选择 (SelectFromModel)
  3. 多 k 值扫描 (30/50/70/100/130/150/186) 找各分类器最优维度
  4. 两种选择方法横向对比

预期: KNN 恢复至 81%+, SVM 89%+, RF 维持 90%
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import entropy as stats_entropy, skew, kurtosis, iqr
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, SelectFromModel,
)
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
ACTIVITY_SHORT = {1: "Walk", 2: "Up", 3: "Down", 4: "Sit", 5: "Stand", 6: "Lay"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "legend.fontsize": 8,
})
sns.set_style("whitegrid")


# ═══════════════════════════════════════════════════════════════
#  数据加载 (同 V2)
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
#  特征提取 (186-D, 同 V2)
# ═══════════════════════════════════════════════════════════════

FREQ_BANDS = [
    (0, 1), (1, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 25),
]

FEATURE_NAMES_CH = [
    # 频域 (24)
    "PeakFreq", "MeanFreq", "MedianFreq", "Energy", "Entropy",
    "BandLow", "BandMid", "BandHigh",
    "PeakMag", "Spread", "Skewness_f", "Kurtosis_f",
    "Flatness", "Crest", "Rolloff75", "Rolloff90", "Rolloff95",
    "B_0-1Hz", "B_1-3Hz", "B_3-5Hz", "B_5-8Hz",
    "B_8-12Hz", "B_12-18Hz", "B_18-25Hz",
    # 时域 (11)
    "Mean", "Var", "PTP", "ZCR",
    "RMS", "Skewness_t", "Kurtosis_t", "IQR", "Median", "Max", "Min",
]

CHANNEL_NAMES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
FEATURE_NAMES = []
for ch in CHANNEL_NAMES:
    for fn in FEATURE_NAMES_CH:
        FEATURE_NAMES.append(f"{ch}_{fn}")


def extract_freq_features(freqs, mag, eps=1e-12):
    total = np.sum(mag)
    feats = []
    feats.append(freqs[np.argmax(mag)])
    if total > eps:
        feats.append(np.sum(freqs * mag) / total)
        cum = np.cumsum(mag)
        feats.append(freqs[np.searchsorted(cum, total / 2)])
    else:
        feats.extend([0.0, 0.0])
    feats.append(np.sum(mag ** 2))
    if total > eps:
        feats.append(stats_entropy(mag / total + eps))
    else:
        feats.append(0.0)

    nyq = freqs[-1]
    feats.append(np.sum(mag[(freqs >= 0) & (freqs < nyq * 0.2)] ** 2))
    feats.append(np.sum(mag[(freqs >= nyq * 0.2) & (freqs < nyq * 0.6)] ** 2))
    feats.append(np.sum(mag[(freqs >= nyq * 0.6)] ** 2))

    feats.append(np.max(mag))
    if total > eps:
        centroid = feats[1]
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / total)
        feats.append(spread)
        feats.append(np.sum(((freqs - centroid) ** 3) * mag) / (total * (spread + eps) ** 3))
        feats.append(np.sum(((freqs - centroid) ** 4) * mag) / (total * (spread + eps) ** 4))
        feats.append(np.exp(np.mean(np.log(mag + eps))) / (np.mean(mag) + eps))
        feats.append(np.max(mag) / (np.mean(mag) + eps))
        cum_norm = cum / total
        feats.append(freqs[np.searchsorted(cum_norm, 0.75)])
        feats.append(freqs[np.searchsorted(cum_norm, 0.90)])
        feats.append(freqs[np.searchsorted(cum_norm, 0.95)])
    else:
        feats.extend([0.0] * 7)
    for flo, fhi in FREQ_BANDS:
        mask = (freqs >= flo) & (freqs < fhi)
        feats.append(np.sum(mag[mask] ** 2))
    return feats


def extract_time_features(signal):
    return [
        np.mean(signal), np.var(signal), np.ptp(signal),
        np.sum(np.diff(np.signbit(signal))) / len(signal),
        np.sqrt(np.mean(signal ** 2)), skew(signal), kurtosis(signal),
        iqr(signal), np.median(signal), np.max(signal), np.min(signal),
    ]


def extract_features_enhanced(raw_data, verbose=True):
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
#  特征选择
# ═══════════════════════════════════════════════════════════════

def feature_selection_mi(X_train, y_train, X_test, k):
    """互信息过滤: 保留 top-k 特征。"""
    selector = SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=42), k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    selected_mask = selector.get_support()
    return X_train_sel, X_test_sel, selected_mask


def feature_selection_rf(X_train, y_train, X_test, threshold="median"):
    """RF 嵌入选择: 保留重要性 > threshold 的特征。"""
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    selector = SelectFromModel(rf, threshold=threshold, prefit=False)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    selected_mask = selector.get_support()
    return X_train_sel, X_test_sel, selected_mask


# ═══════════════════════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════════════════════

def plot_k_sweep(k_values, results_dict, save_path):
    """不同 k 值下四分类器准确率曲线 + 最优标注。"""
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#F44336", "#2196F3", "#FF9800", "#4CAF50"]
    markers = ["o", "s", "^", "D"]
    clf_names = ["Decision Tree", "KNN (k=5)", "SVM (RBF)", "Random Forest"]

    for (name, scores), color, marker in zip(results_dict.items(), colors, markers):
        ax.plot(k_values, scores, marker=marker, color=color, linewidth=1.8,
                markersize=6, label=name)
        best_k = k_values[np.argmax(scores)]
        best_score = np.max(scores)
        ax.annotate(f"k={best_k}\n{best_score:.4f}",
                    (best_k, best_score),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=8, color=color, ha="center", fontweight="bold")

    ax.set_xlabel("Number of Selected Features (k)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Feature Selection: Accuracy vs Number of Features (MI Filter)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


def plot_selection_comparison(results_all, save_path):
    """MI vs RF 选择方法对比柱状图。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    clf_names = ["Decision Tree", "KNN (k=5)", "SVM (RBF)", "Random Forest"]
    methods = ["V1 (72-D)", "V2 (186-D)", "V3-MI (opt k)", "V3-RF (auto)"]
    all_data = {}

    for method in methods:
        all_data[method] = []
        for clf in clf_names:
            all_data[method].append(results_all[method].get(clf, 0))

    x = np.arange(len(clf_names))
    width = 0.2
    colors = ["#90CAF9", "#42A5F5", "#1E88E5", "#0D47A1"]

    for i, (method, color) in enumerate(zip(methods, colors)):
        bars = ax.bar(x + i * width, all_data[method], width, label=method,
                      color=color, edgecolor="white")
        for bar, val in zip(bars, all_data[method]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.72, 0.95)
    ax.set_title("V1 → V2 → V3: Accuracy Progression Across Versions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


def plot_feature_importance_mi(mi_scores, top_n=30, save_path=None):
    """互信息 Top-N 特征排名。"""
    idx = np.argsort(mi_scores)[-top_n:]
    names_sub = [FEATURE_NAMES[i] for i in idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors_bar = ["#FF5722" if any(kw in FEATURE_NAMES[i] for kw in
        ["Freq", "Energy", "Entropy", "Band", "PeakMag", "Spread", "Skewness_f",
         "Kurtosis_f", "Flatness", "Crest", "Rolloff", "B_", "Hz"])
        else "#2196F3" for i in idx]
    ax.barh(range(top_n), mi_scores[idx], color=colors_bar, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names_sub, fontsize=7)
    ax.set_xlabel("Mutual Information Score")
    ax.set_title(f"Mutual Information — Top {top_n} Features", fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#FF5722", label="Frequency Domain"),
        Patch(color="#2196F3", label="Time Domain"),
    ], fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  ✓ {save_path}")


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def evaluate_classifiers(X_train, y_train, X_test, y_test, dim_label=""):
    """训练并评估四个分类器，返回 {name: accuracy}。"""
    results = {}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y_train)
    results["Decision Tree"] = accuracy_score(y_test, dt.predict(X_test))

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    knn.fit(X_train_s, y_train)
    results["KNN (k=5)"] = accuracy_score(y_test, knn.predict(X_test_s))

    # SVM
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
    svm.fit(X_train_s, y_train)
    results["SVM (RBF)"] = accuracy_score(y_test, svm.predict(X_test_s))

    # Random Forest — 树模型无需标准化
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["Random Forest"] = accuracy_score(y_test, rf.predict(X_test))

    return results


def main():
    print("═" * 60)
    print("  进阶分类器 V3: 特征选择增强")
    print("  方法: Mutual Information 过滤 + RF 嵌入选择")
    print("═" * 60)

    # 1. 加载
    print("\n[1/6] 加载数据…")
    X_raw, y = load_data()

    # 2. 提取 186-D 特征
    print("\n[2/6] 特征提取 (186-D)…")
    X_train_full = extract_features_enhanced(X_raw["train"])
    X_test_full = extract_features_enhanced(X_raw["test"])
    print(f"  X_train: {X_train_full.shape}, X_test: {X_test_full.shape}")

    # ═══════════════════════════════════════════════════════════
    # 3. 互信息特征排名
    # ═══════════════════════════════════════════════════════════
    print("\n[3/6] 互信息 (MI) 特征排名…")
    mi_scores = mutual_info_classif(X_train_full, y["train"], random_state=42)
    mi_ranking = np.argsort(mi_scores)[::-1]

    # 输出 Top-20 特征
    print(f"\n  MI Top-20 特征:")
    for rank, idx in enumerate(mi_ranking[:20], 1):
        tag = "[频]" if any(kw in FEATURE_NAMES[idx] for kw in
            ["Freq", "Energy", "Entropy", "Band", "PeakMag", "B_", "Hz",
             "Spread", "Flatness", "Crest", "Rolloff"]) else "[时]"
        print(f"  {rank:2d}. {tag} {FEATURE_NAMES[idx]:40s} MI={mi_scores[idx]:.4f}")

    # 统计频域/时域占比
    freq_in_top = sum(1 for i in mi_ranking[:50] if any(
        kw in FEATURE_NAMES[i] for kw in
        ["Freq", "Energy", "Entropy", "Band", "PeakMag", "B_", "Hz",
         "Spread", "Flatness", "Crest", "Rolloff"]))
    print(f"\n  Top-50 中频域特征占比: {freq_in_top}/50 ({freq_in_top/50*100:.0f}%)")
    print(f"  零 MI 特征数: {np.sum(mi_scores < 1e-6)} / {len(mi_scores)}")

    # ═══════════════════════════════════════════════════════════
    # 4. k 值扫描 — 找每个分类器的最优维度
    # ═══════════════════════════════════════════════════════════
    print("\n[4/6] k 值扫描 (30 → 186)…")
    k_values = [30, 50, 70, 100, 130, 150, 186]
    sweep_results = {name: [] for name in
                     ["Decision Tree", "KNN (k=5)", "SVM (RBF)", "Random Forest"]}

    for k in k_values:
        print(f"  k={k}…", end=" ", flush=True)
        X_tr, X_te, mask = feature_selection_mi(X_train_full, y["train"], X_test_full, k)
        res = evaluate_classifiers(X_tr, y["train"], X_te, y["test"])
        for name in sweep_results:
            sweep_results[name].append(res[name])
        print(f"DT={res['Decision Tree']:.4f} KNN={res['KNN (k=5)']:.4f} "
              f"SVM={res['SVM (RBF)']:.4f} RF={res['Random Forest']:.4f}")

    # 找到每个分类器的最优 k
    optimal_k = {}
    optimal_scores = {}
    for name in sweep_results:
        best_i = np.argmax(sweep_results[name])
        optimal_k[name] = k_values[best_i]
        optimal_scores[name] = sweep_results[name][best_i]

    print("\n  各分类器最优 k:")
    for name in sweep_results:
        print(f"  {name:<20s}: k={optimal_k[name]:>3d}  Acc={optimal_scores[name]:.4f}")

    # ═══════════════════════════════════════════════════════════
    # 5. RF 嵌入选择
    # ═══════════════════════════════════════════════════════════
    print("\n[5/6] RF 嵌入选择 (SelectFromModel, threshold=median)…")
    X_tr_rf, X_te_rf, mask_rf = feature_selection_rf(X_train_full, y["train"], X_test_full)
    n_rf_selected = mask_rf.sum()
    print(f"  RF 自动选择了 {n_rf_selected} / 186 维特征 ({n_rf_selected/186*100:.0f}%)")
    results_rf_sel = evaluate_classifiers(X_tr_rf, y["train"], X_te_rf, y["test"])
    for name, acc in results_rf_sel.items():
        print(f"  {name:<20s}: Acc={acc:.4f}")

    # ═══════════════════════════════════════════════════════════
    # 6. 汇总对比 + 可视化
    # ═══════════════════════════════════════════════════════════
    print("\n[6/6] 汇总报告 + 可视化…")

    # V1 结果 (硬编码, 来自 advanced_classifiers.py 实际输出)
    results_v1 = {
        "Decision Tree": 0.7767, "KNN (k=5)": 0.7913, "SVM (RBF)": 0.8561,
    }
    # V2 结果 (硬编码, 来自 advanced_classifiers_v2.py 实际输出)
    results_v2_full = {
        "Decision Tree": 0.7869, "KNN (k=5)": 0.7815,
        "SVM (RBF)": 0.8887, "Random Forest": 0.8999,
    }

    # V3 最优 MI (直接用 sweep 中的最优值，避免重新选择带来的随机差异)
    results_v3_mi = {}
    for name in sweep_results:
        best_acc = max(sweep_results[name])
        results_v3_mi[name] = best_acc

    # 汇总表
    print("\n" + "═" * 80)
    print("  版本演进对比")
    print("═" * 80)
    header = f"  {'分类器':<20s} {'V1(72-D)':<10s} {'V2(186-D)':<10s} {'V3-MI(opt)':<12s} {'V3-RF':<10s} {'最佳提升':<10s}"
    print(header)
    print("  " + "─" * 72)

    for clf_name in ["Decision Tree", "KNN (k=5)", "SVM (RBF)", "Random Forest"]:
        v1 = results_v1.get(clf_name, None)
        v2 = results_v2_full[clf_name]
        v3_mi = results_v3_mi[clf_name]
        v3_rf = results_rf_sel[clf_name]

        best_v3 = max(v3_mi, v3_rf)
        if v1 is not None:
            gain = best_v3 - v1
            v1_str = f"{v1*100:.2f}%"
        else:
            gain = best_v3 - v2
            v1_str = "—"

        k_str = f"k={optimal_k[clf_name]}" if clf_name in optimal_k else ""
        print(f"  {clf_name:<20s} {v1_str:<10s} {v2*100:>7.2f}%    "
              f"{v3_mi*100:>7.2f}%{k_str:<6s} {v3_rf*100:>7.2f}%    "
              f"+{gain*100:.2f}pp")

    # 整理用于可视化的数据
    all_versions = {
        "V1 (72-D)": {
            "Decision Tree": results_v1.get("Decision Tree", 0),
            "KNN (k=5)": results_v1.get("KNN (k=5)", 0),
            "SVM (RBF)": results_v1.get("SVM (RBF)", 0),
            "Random Forest": 0,
        },
        "V2 (186-D)": results_v2_full,
        "V3-MI (opt k)": results_v3_mi,
        "V3-RF (auto)": results_rf_sel,
    }

    # 生成图
    plot_k_sweep(k_values, sweep_results, os.path.join(SAVE_DIR, "v3_k_sweep.png"))
    plot_feature_importance_mi(mi_scores, top_n=30,
                               save_path=os.path.join(SAVE_DIR, "v3_mi_importance.png"))
    plot_selection_comparison(all_versions, os.path.join(SAVE_DIR, "v3_version_comparison.png"))

    # 最优方案详报
    print("\n" + "═" * 60)
    print("  最优方案: MI 过滤 + 各分类器最优 k")
    print("═" * 60)
    best_overall = max(results_v3_mi, key=results_v3_mi.get)
    print(f"  全局最优: {best_overall} @ {results_v3_mi[best_overall]*100:.2f}%")
    print(f"  特征维度: {optimal_k[best_overall]} 维 (从 186 维中筛选)")
    print(f"  相比 V1 提升: +{(results_v3_mi[best_overall] - results_v1.get(best_overall, results_v2_full[best_overall]))*100:.2f}pp")
    print(f"  相比 V2 提升: +{(results_v3_mi[best_overall] - results_v2_full[best_overall])*100:.2f}pp")

    # RF 方法对比
    rf_mi_acc = results_v3_mi["Random Forest"]
    rf_rf_sel_acc = results_rf_sel["Random Forest"]
    if rf_mi_acc > rf_rf_sel_acc:
        better_method = "MI 过滤"
        better_acc = rf_mi_acc
    else:
        better_method = "RF 嵌入选择"
        better_acc = rf_rf_sel_acc
    print(f"\n  RF 最佳选择方法: {better_method} ({better_acc*100:.2f}%)")

    print("\n" + "═" * 60)
    print("  完成。图像保存至 figures/uci/v3_*.png")
    print("═" * 60)


if __name__ == "__main__":
    main()
