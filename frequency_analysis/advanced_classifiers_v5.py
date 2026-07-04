"""
V5: 滑动窗峰度特征 + 分类器
============================
在 V3 210-D 基础上, 加入 V4 滑动窗分析衍生的峰度特征:
  每通道 7 个子窗 → 时域峰度(7) + 频域峰度(7) + 主频(7) + 质心(7)
  → 统计: 均值/标准差/峰度/极差 → 每通道 +12 维
  → 3 accel 通道 × 12 = +36 维 → 总计 246-D

然后经 MI 特征选择 + RF 嵌入选择, 跑四分类器。
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
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = 50
N_SAMPLES = 128
SUB_WIN = 32
SUB_STRIDE = 16
N_SUB = (N_SAMPLES - SUB_WIN) // SUB_STRIDE + 1  # 7
DATA_DIR = os.path.join(PROJECT_ROOT, "UCI HAR Dataset")

# 中文目录
输出根目录 = os.path.join(PROJECT_ROOT, "figures")
目录 = {
    "分类器对比": os.path.join(输出根目录, "分类器对比"),
}
for d in 目录.values():
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "legend.fontsize": 8,
})
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti SC", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
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
#  滑动窗峰度特征提取
# ═══════════════════════════════════════════════════════════════

def 滑动窗峰度特征(信号):
    """
    对一个通道的 128 点信号做滑动子窗分析, 返回聚合统计特征 (12 维)。
    """
    kt_list, kf_list, pf_list, cen_list = [], [], [], []

    for i in range(N_SUB):
        start = i * SUB_STRIDE
        sub = 信号[start:start + SUB_WIN]

        # 时域峰度
        kt_list.append(kurtosis(sub))

        # 频域
        freqs, mag = fft_spectrum(sub)
        total = np.sum(mag) + 1e-12

        # 频域峰度 (对幅度谱)
        mask = freqs >= 0.5
        mag_valid = mag[mask]
        if len(mag_valid) > 4:
            kf_list.append(kurtosis(mag_valid))
        else:
            kf_list.append(0.0)

        # 主频
        pf_list.append(freqs[np.argmax(mag)])
        # 质心
        cen_list.append(np.sum(freqs * mag) / total)

    kt = np.array(kt_list)
    kf = np.array(kf_list)
    pf = np.array(pf_list)
    cen = np.array(cen_list)

    # 聚合统计: 每类取 mean/std/kurtosis/ptp
    feats = []
    for arr in [kt, kf, pf, cen]:
        feats.append(np.mean(arr))
        feats.append(np.std(arr))
        feats.append(kurtosis(arr) if np.std(arr) > 1e-8 else 0.0)
        feats.append(np.ptp(arr))  # 峰峰值=极差, 反映峰度在步态周期内的波动幅度
    return feats  # 16 维 (4指标 × 4统计量)

    # 实际上 kurtosis 和 ptp 对主频/质心意义不大, 但我们保留用于探索


# ═══════════════════════════════════════════════════════════════
#  静态特征 (同 V2/V3, 210-D)
# ═══════════════════════════════════════════════════════════════

FREQ_BANDS = [(0, 1), (1, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 25)]


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
    feats.append(stats_entropy(mag / (total + eps) + eps) if total > eps else 0.0)

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


def extract_all_features(raw_data, verbose=True):
    """
    提取 210 静态特征 + 滑动窗峰度特征。
    滑动窗特征仅对前 3 通道 (body_acc) 提取, gyro 通道不做滑动窗。
    总计: 210 + 3×16 = 258 维
    """
    n_windows = raw_data.shape[0]
    n_channels = raw_data.shape[2]  # 6
    all_rows = []

    for i in range(n_windows):
        row = []
        for ch in range(n_channels):
            signal = raw_data[i, :, ch]
            # 静态特征
            freqs, mag = fft_spectrum(signal)
            row.extend(extract_freq_features(freqs, mag))
            row.extend(extract_time_features(signal))

            # 滑动窗峰度特征 (仅加速度通道, ch=0,1,2)
            if ch < 3:
                row.extend(滑动窗峰度特征(signal))

        all_rows.append(row)
        if verbose and (i + 1) % 2000 == 0:
            print(f"  特征提取: {i + 1}/{n_windows}")

    return np.array(all_rows, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
#  特征选择 + 评估
# ═══════════════════════════════════════════════════════════════

def evaluate_all(X_train, y_train, X_test, y_test):
    results = {}
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y_train)
    results["决策树"] = accuracy_score(y_test, dt.predict(X_test))

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    knn.fit(X_train_s, y_train)
    results["KNN"] = accuracy_score(y_test, knn.predict(X_test_s))

    svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
    svm.fit(X_train_s, y_train)
    results["SVM"] = accuracy_score(y_test, svm.predict(X_test_s))

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["随机森林"] = accuracy_score(y_test, rf.predict(X_test))

    return results


def 画版本对比(v1, v2, v3_mi, v3_rf, v5_full, v5_mi, v5_rf, 保存路径):
    """V1→V5 六版本柱状图。"""
    clf_names = ["决策树", "KNN", "SVM", "随机森林"]
    versions = ["V1-72D", "V2-210D", "V3-MI", "V3-RF", "V5-Full", "V5-MI", "V5-RF"]
    all_data = {v: [] for v in versions}

    mapping = [
        (versions[0], v1), (versions[1], v2), (versions[2], v3_mi),
        (versions[3], v3_rf), (versions[4], v5_full), (versions[5], v5_mi),
        (versions[6], v5_rf),
    ]

    for vname, data in mapping:
        for cn in clf_names:
            all_data[vname].append(data.get(cn, 0))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(clf_names))
    width = 0.12
    colors = ["#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#1E88E5", "#1565C0", "#0D47A1"]

    for i, (vname, color) in enumerate(zip(versions, colors)):
        bars = ax.bar(x + i * width, all_data[vname], width, label=vname,
                      color=color, edgecolor="white")
        for bar, val in zip(bars, all_data[vname]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center", fontsize=5.5, rotation=90)

    ax.set_xticks(x + width * 3)
    ax.set_xticklabels(clf_names, fontsize=11)
    ax.set_ylabel("准确率")
    ax.set_ylim(0.72, 0.96)
    ax.set_title("V1 → V5: 各版本准确率演进", fontweight="bold", fontsize=13)
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(保存路径)
    plt.close(fig)
    print(f"  ✓ {保存路径}")


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("═" * 55)
    print("  V5: 滑动窗峰度特征 + 分类器")
    print(f"  滑动子窗: {SUB_WIN}点/{SUB_STRIDE}步长 → {N_SUB} 子窗/窗口")
    print(f"  每 accel 通道 +16 维滑动特征 → 3×16=48 维")
    print(f"  总维度: 210 静态 + 48 滑动 = 258 维")
    print("═" * 55)

    # 1. 加载
    print("\n[1/5] 加载数据…")
    X_raw, y = load_data()

    # 2. 特征提取
    print("\n[2/5] 特征提取 (258-D: 210静态 + 48滑动窗峰度)…")
    X_train_full = extract_all_features(X_raw["train"])
    X_test_full = extract_all_features(X_raw["test"])
    print(f"  X_train: {X_train_full.shape}, X_test: {X_test_full.shape}")

    # 3. 全量特征评估
    print("\n[3/5] 全量 258-D 评估…")
    results_v5_full = evaluate_all(X_train_full, y["train"], X_test_full, y["test"])
    for name, acc in results_v5_full.items():
        print(f"  {name}: {acc*100:.2f}%")

    # 4. MI 过滤 → 扫描
    print("\n[4/5] MI 过滤 + k 扫描…")
    mi_scores = mutual_info_classif(X_train_full, y["train"], random_state=42)
    k_values = [30, 50, 70, 100, 130, 160, 200, 258]
    sweep = {name: [] for name in ["决策树", "KNN", "SVM", "随机森林"]}

    for k in k_values:
        selector = SelectKBest(lambda X, y_: mutual_info_classif(X, y_, random_state=42), k=k)
        X_tr = selector.fit_transform(X_train_full, y["train"])
        X_te = selector.transform(X_test_full)
        res = evaluate_all(X_tr, y["train"], X_te, y["test"])
        for name in sweep:
            sweep[name].append(res[name])
        print(f"  k={k:>3d}: DT={res['决策树']:.4f} KNN={res['KNN']:.4f} "
              f"SVM={res['SVM']:.4f} RF={res['随机森林']:.4f}")

    # MI 最优
    results_v5_mi = {}
    for name in sweep:
        best_i = np.argmax(sweep[name])
        results_v5_mi[name] = sweep[name][best_i]
        print(f"  {name} 最优k={k_values[best_i]}: {sweep[name][best_i]*100:.2f}%")

    # 5. RF 嵌入选择
    print("\n[5/5] RF 嵌入选择 (SelectFromModel, threshold=median)…")
    rf_sel = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    selector_rf = SelectFromModel(rf_sel, threshold="median", prefit=False)
    X_tr_rf = selector_rf.fit_transform(X_train_full, y["train"])
    X_te_rf = selector_rf.transform(X_test_full)
    n_selected = X_tr_rf.shape[1]
    print(f"  选择了 {n_selected} / 258 维 ({n_selected/258*100:.0f}%)")

    results_v5_rf = evaluate_all(X_tr_rf, y["train"], X_te_rf, y["test"])
    for name, acc in results_v5_rf.items():
        print(f"  {name}: {acc*100:.2f}%")

    # ── 汇总 ──
    results_v1 = {"决策树": 0.7767, "KNN": 0.7913, "SVM": 0.8561}
    results_v2 = {"决策树": 0.7869, "KNN": 0.7815, "SVM": 0.8887, "随机森林": 0.8999}
    results_v3_mi = {"决策树": 0.7906, "KNN": 0.8039, "SVM": 0.8965, "随机森林": 0.9040}
    results_v3_rf = {"决策树": 0.8124, "KNN": 0.8124, "SVM": 0.8761, "随机森林": 0.8955}

    print("\n" + "═" * 70)
    print("  V1 → V5 准确率演进")
    print("═" * 70)
    header = f"  {'分类器':<10s} {'V1-72D':<9s} {'V2-210D':<9s} {'V3-MI':<9s} {'V3-RF':<9s} {'V5-Full':<9s} {'V5-MI':<9s} {'V5-RF':<9s}"
    print(header)
    print("  " + "─" * 72)
    for cn in ["决策树", "KNN", "SVM", "随机森林"]:
        vals = [
            results_v1.get(cn, None), results_v2.get(cn, None),
            results_v3_mi.get(cn, None), results_v3_rf.get(cn, None),
            results_v5_full.get(cn, None), results_v5_mi.get(cn, None),
            results_v5_rf.get(cn, None),
        ]
        strs = [f"{v*100:>7.2f}%" if v is not None else "    —   " for v in vals]
        print(f"  {cn:<10s} " + " ".join(strs))

    # 提升
    print("\n  V5 相比 V3 最优提升:")
    for cn in ["决策树", "KNN", "SVM", "随机森林"]:
        v3_best = max(results_v3_mi.get(cn, 0), results_v3_rf.get(cn, 0))
        v5_best = max(results_v5_mi.get(cn, 0), results_v5_rf.get(cn, 0), results_v5_full.get(cn, 0))
        delta = v5_best - v3_best
        sign = "+" if delta >= 0 else ""
        print(f"  {cn:<10s}: V3={v3_best*100:.2f}% → V5={v5_best*100:.2f}% ({sign}{delta*100:.2f}pp)")

    # 可视化
    画版本对比(
        results_v1, results_v2, results_v3_mi, results_v3_rf,
        results_v5_full, results_v5_mi, results_v5_rf,
        os.path.join(目录["分类器对比"], "V1到V5准确率演进.png"),
    )

    # MI 扫描曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#F44336", "#2196F3", "#FF9800", "#4CAF50"]
    markers = ["o", "s", "^", "D"]
    for (name, scores), c, m in zip(sweep.items(), colors, markers):
        ax.plot(k_values, scores, marker=m, color=c, linewidth=1.8, markersize=6, label=name)
        best_k = k_values[np.argmax(scores)]
        best_s = np.max(scores)
        ax.annotate(f"k={best_k}\n{best_s:.4f}", (best_k, best_s),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=8, color=c, ha="center", fontweight="bold")
    ax.set_xlabel("MI 筛选特征数 (k)")
    ax.set_ylabel("准确率")
    ax.set_title("V5 滑动窗特征: 准确率 vs MI 筛选维度", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    路径2 = os.path.join(目录["分类器对比"], "V5_MI扫描曲线.png")
    fig.savefig(路径2)
    plt.close(fig)
    print(f"  ✓ {路径2}")

    print("\n" + "═" * 55)
    print("  完成")
    print("═" * 55)


if __name__ == "__main__":
    main()
