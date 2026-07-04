"""
信号与系统 — 运动传感器时频域分析展示专项
==========================================
单一精简脚本，完成全流程：
  时域波形 → STFT 时频分析 → FFT 频谱对比 → 频带能量
  → 特征提取(时域+频域) → 决策树分类 → 可视化评估

数据集: UCI HAR Dataset (50 Hz, 128 点/窗, 6 类动作)
输出: showcase/figures/ 下 7 张图 + 终端分类报告
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
from scipy.stats import entropy as stats_entropy
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# ═══════════════════════════════════════════════════════════════
#  全局配置
# ═══════════════════════════════════════════════════════════════

FS = 50                              # 采样率 (Hz)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "UCI HAR Dataset")
SAVE_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)

ACTIVITY = {1: "行走", 2: "上楼", 3: "下楼", 4: "静坐", 5: "站立", 6: "躺卧"}
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
ACT_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]
BAND_COLORS = ["#1B5E20", "#4CAF50", "#FFC107", "#F44336"]

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "font.sans-serif": ["Arial Unicode MS", "SimHei", "Heiti SC", "STHeiti"],
    "axes.unicode_minus": False,
})
sns.set_style("whitegrid")


# ═══════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════

def load_uci():
    """加载 UCI HAR 训练集 & 测试集的原始惯性信号。"""
    X, y = {}, {}
    for subset in ["train", "test"]:
        path = os.path.join(DATA_DIR, subset, "Inertial Signals")
        channels = []
        for axis in ["x", "y", "z"]:
            for signal in ["body_acc", "body_gyro"]:
                data = np.loadtxt(os.path.join(path, f"{signal}_{axis}_{subset}.txt"))
                channels.append(data)
        X[subset] = np.stack(channels, axis=-1).astype(np.float32)
        y[subset] = np.loadtxt(
            os.path.join(DATA_DIR, subset, f"y_{subset}.txt")).astype(int)
    return X, y


# ═══════════════════════════════════════════════════════════════
#  数字信号处理工具
# ═══════════════════════════════════════════════════════════════

def fft_spectrum(signal):
    """单边幅度谱。输入 (N,) → (freqs, mag)。"""
    n = len(signal)
    mag = np.abs(fft(signal)) / n
    mag = mag[: n // 2 + 1]
    mag[1:-1] *= 2
    freqs = fftfreq(n, 1 / FS)[: n // 2 + 1]
    return freqs, mag


def extract_time_features(signal):
    """提取 4 个时域特征 (返回 list 与 freq 特征保持一致)。"""
    return [
        np.mean(signal),
        np.var(signal),
        np.ptp(signal),
        np.sum(np.diff(np.signbit(signal))) / len(signal),
    ]


# 频域 & 时域特征名列表
FREQ_FEATURE_KEYS = [
    "主峰频率", "均值频率", "中位数频率", "频谱能量", "谱熵",
    "低带能量(0-5Hz)", "中带能量(5-15Hz)", "高带能量(15-25Hz)",
]
TIME_FEATURE_KEYS = ["均值", "方差", "峰峰值", "过零率"]


def extract_freq_features(mag, freqs):
    """从单边幅度谱提取 8 个频域特征。"""
    total = np.sum(mag) + 1e-12
    nyq = FS / 2
    cum = np.cumsum(mag)
    med_idx = min(np.searchsorted(cum, total / 2), len(freqs) - 1)
    return [
        freqs[np.argmax(mag)],                                   # 主峰频率
        np.sum(freqs * mag) / total,                             # 均值频率
        freqs[med_idx],                                          # 中位数频率
        np.sum(mag ** 2),                                        # 频谱能量
        stats_entropy(mag / total + 1e-12),                      # 谱熵
        np.sum(mag[(freqs >= 0) & (freqs < 5)] ** 2),            # 低带 0-5Hz
        np.sum(mag[(freqs >= 5) & (freqs < 15)] ** 2),           # 中带 5-15Hz
        np.sum(mag[(freqs >= 15) & (freqs <= nyq)] ** 2),        # 高带 15-25Hz
    ]


def build_feature_matrix(raw_data):
    """逐窗口逐通道提取时域+频域特征 → (n_windows, 72) 特征矩阵。"""
    n_win, _, n_ch = raw_data.shape
    rows = []
    for i in range(n_win):
        row = []
        for ch in range(n_ch):
            freqs, mag = fft_spectrum(raw_data[i, :, ch])
            row.extend(extract_freq_features(mag, freqs))
            row.extend(extract_time_features(raw_data[i, :, ch]))
        rows.append(row)
    names = [f"{ch}_{k}" for ch in CHANNELS
             for k in FREQ_FEATURE_KEYS + TIME_FEATURE_KEYS]
    return np.array(rows), names


# ═══════════════════════════════════════════════════════════════
#  图 1: 六种动作时域波形
# ═══════════════════════════════════════════════════════════════

def plot_waveforms(X_train, y_train):
    """每类取一个代表性窗口，绘制加速度计三轴时域波形。"""
    t = np.arange(128) / FS
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for act_id in range(1, 7):
        ax = axes[(act_id - 1) // 3][(act_id - 1) % 3]
        idxs = np.where(y_train == act_id)[0]
        win = X_train[idxs[len(idxs) // 2]]
        for ch, color, label in [(0, "#F44336", "X"), (1, "#4CAF50", "Y"), (2, "#2196F3", "Z")]:
            ax.plot(t, win[:, ch], color=color, linewidth=0.8, label=label, alpha=0.85)
        ax.set_title(f"活动 {act_id}: {ACTIVITY[act_id]}", fontsize=10)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("加速度 (g)")
        ax.set_ylim(-5, 15)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("图 1: 六种动作加速度计三轴时域波形 (红=X 绿=Y 蓝=Z)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "01_时域波形对比.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  图 2: STFT 频带能量时间演进 (折线图替代热力图)
# ═══════════════════════════════════════════════════════════════

def plot_stft_band_energy(X_train, y_train):
    """
    对行走 vs 静坐的单窗口做 STFT，按频带聚合能量，
    在时间轴上画折线，直接可读可对比。
    """
    bands = [("DC–2 Hz", 0, 2), ("2–5 Hz", 2, 5),
             ("5–10 Hz", 5, 10), ("10–25 Hz", 10, 25)]
    pairs = [((1, "行走"), (4, "静坐")), ((1, "行走"), (6, "躺卧"))]
    channels = [(0, "acc_x"), (2, "acc_z")]

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    for row, ((a_id, a_name), (b_id, b_name)) in enumerate(pairs):
        for col, (ch_idx, ch_name) in enumerate(channels):
            ax = fig.add_subplot(gs[row, col])
            for act_id, name, ls, lw in [(a_id, a_name, "-", 1.8), (b_id, b_name, "--", 1.5)]:
                idxs = np.where(y_train == act_id)[0]
                win = X_train[idxs[len(idxs) // 2], :, ch_idx]
                f_vals, t_vals, Sxx = spectrogram(win, fs=FS, nperseg=32, noverlap=24, nfft=64)
                for band_name, lo, hi in bands:
                    e = np.sum(Sxx[(f_vals >= lo) & (f_vals < hi), :], axis=0)
                    ax.plot(t_vals, e, color=ACT_COLORS[act_id - 1], linestyle=ls,
                            linewidth=lw, alpha=0.85,
                            label=f"{name} {band_name}" if col == 0 else "")
            ax.set_title(f"{a_name} vs {b_name} — {ch_name}", fontsize=10)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("频带能量")
            ax.set_xlim(t_vals[0], t_vals[-1])

    # 统一图例
    handles, labels = ax.get_legend_handles_labels()
    leg_ax = fig.add_axes([0.1, 0.01, 0.8, 0.04])
    leg_ax.set_axis_off()
    leg_ax.legend(handles, labels, loc="center", ncol=4, fontsize=7,
                  title="实线=行走  虚线=静态", title_fontsize=8)

    fig.suptitle("图 2: STFT 频带能量时间演进 — 折线图 (替代传统热力图)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.savefig(os.path.join(SAVE_DIR, "02_STFT频带能量时间演进.png"), bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  图 3: FFT 平均幅度谱叠加对比
# ═══════════════════════════════════════════════════════════════

def plot_fft_overlay(X_train, y_train):
    """六种动作的平均 FFT 幅度谱，在 acc_x/y/z 三轴分别叠加。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for col, (ch_idx, ch_name) in enumerate([(0, "acc_x"), (1, "acc_y"), (2, "acc_z")]):
        ax = axes[col]
        for act_id in range(1, 7):
            mask = y_train == act_id
            windows = X_train[mask, :, ch_idx]
            all_mags = np.array([fft_spectrum(w)[1] for w in windows])
            mean_mag = all_mags.mean(axis=0)
            freqs, _ = fft_spectrum(X_train[0, :, ch_idx])
            mask_f = freqs <= 25
            ax.plot(freqs[mask_f], mean_mag[mask_f], color=ACT_COLORS[act_id - 1],
                    linewidth=1.3, alpha=0.85, label=ACTIVITY[act_id])
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("平均幅度")
        ax.set_title(ch_name, fontsize=10)
        ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("图 3: FFT 平均幅度谱叠加对比 — 六种动作 × 三轴",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "03_FFT平均幅度谱对比.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  图 4: 频带能量堆叠柱状图
# ═══════════════════════════════════════════════════════════════

def plot_band_energy(X_train, y_train):
    """六种动作在 5 个频段上的归一化能量占比 (acc_x, acc_z)。"""
    bands = [("DC–2Hz", 0, 2), ("2–5Hz", 2, 5), ("5–10Hz", 5, 10),
             ("10–15Hz", 10, 15), ("15–25Hz", 15, 25)]
    band_colors = ["#1B5E20", "#4CAF50", "#FFC107", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ch_idx, ch_name in [(axes[0], 0, "acc_x"), (axes[1], 2, "acc_z")]:
        pcts = []
        for act_id in range(1, 7):
            mask = y_train == act_id
            band_e = np.zeros(len(bands))
            for i in np.where(mask)[0]:
                freqs, mag = fft_spectrum(X_train[i, :, ch_idx])
                for b_idx, (_, lo, hi) in enumerate(bands):
                    band_e[b_idx] += np.sum(mag[(freqs >= lo) & (freqs < hi)] ** 2)
            pcts.append(band_e / band_e.sum() * 100)
        pcts = np.array(pcts)
        bottom = np.zeros(6)
        x = np.arange(6)
        for b_idx in range(len(bands)):
            ax.bar(x, pcts[:, b_idx], bottom=bottom, color=band_colors[b_idx],
                   edgecolor="white", linewidth=0.5, label=bands[b_idx][0])
            bottom += pcts[:, b_idx]
        ax.set_xticks(x)
        ax.set_xticklabels([ACTIVITY[i] for i in range(1, 7)], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("能量占比 (%)")
        ax.set_title(f"{ch_name} 频带能量分布", fontsize=10)
        ax.legend(fontsize=7, ncol=5, loc="upper right")
        ax.set_ylim(0, 105)
    fig.suptitle("图 4: 频带能量堆叠柱状图 — 六种动作 (acc_x / acc_z)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "04_频带能量分布.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  图 5: 频谱特征空间散点图
# ═══════════════════════════════════════════════════════════════

def plot_spectral_scatter(X_train, y_train):
    """主峰值频率 vs 谱熵 / 谱能量 — 展示频域空间中的类可分性。"""
    rng = np.random.RandomState(42)
    idxs = rng.choice(len(y_train), min(1500, len(y_train)), replace=False)

    pf, ent, eng, labs = [], [], [], []
    for i in idxs:
        freqs, mag = fft_spectrum(X_train[i, :, 0])
        total = np.sum(mag) + 1e-12
        pf.append(freqs[np.argmax(mag)])
        ent.append(stats_entropy(mag / total + 1e-12))
        eng.append(np.sum(mag ** 2))
        labs.append(y_train[i])

    pf, ent, eng, labs = np.array(pf), np.array(ent), np.array(eng), np.array(labs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, y_vals, y_label, log_scale in [
        (axes[0], ent, "谱熵 (bits)", False),
        (axes[1], eng, "频谱能量", True),
    ]:
        for act_id in range(1, 7):
            mask = labs == act_id
            ax.scatter(pf[mask], y_vals[mask], s=6, alpha=0.5,
                       color=ACT_COLORS[act_id - 1], label=ACTIVITY[act_id])
        ax.set_xlabel("主峰值频率 (Hz)")
        ax.set_ylabel(y_label)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(fontsize=7, markerscale=2.5)
    axes[0].set_title("峰值频率 vs 谱熵")
    axes[1].set_title("峰值频率 vs 频谱能量 (对数坐标)")
    fig.suptitle("图 5: 频谱特征空间散点图 — acc_x, 1500 样本",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "05_频谱特征散点图.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  图 6: 特征重要性排名
# ═══════════════════════════════════════════════════════════════

def plot_feature_importance(importances, feat_names):
    """Top-20 特征重要性，红色=频域，蓝色=时域。"""
    top_n = 20
    idx = np.argsort(importances)[::-1][:top_n]
    is_freq = np.array([n.split("_")[-1] in FREQ_FEATURE_KEYS
                        for n in np.array(feat_names)[idx]])
    colors = ["#FF5722" if f else "#2196F3" for f in is_freq]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(range(top_n), importances[idx][::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(np.array(feat_names)[idx][::-1], fontsize=7.5)
    ax.set_xlabel("Gini 重要性")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#FF5722", label="频域特征"), Patch(color="#2196F3", label="时域特征")
    ], fontsize=9, loc="lower right")
    ax.set_title("图 6: 特征重要性 Top-20 (红=频域  蓝=时域)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "06_特征重要性排名.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  图 7: 混淆矩阵
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred):
    """测试集混淆矩阵。"""
    cm = confusion_matrix(y_true, y_pred)
    labels = [ACTIVITY[i] for i in range(1, 7)]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, annot_kws={"fontsize": 10})
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title("图 7: 测试集混淆矩阵", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "07_混淆矩阵.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("═" * 55)
    print("  信号与系统 — 运动传感器时频域分析展示")
    print("═" * 55)

    # 1. 加载数据
    print("\n[1/5] 加载 UCI HAR 数据集…")
    X, y = load_uci()
    print(f"  训练: {X['train'].shape[0]} 窗口, 测试: {X['test'].shape[0]} 窗口")
    print(f"  类别: {[ACTIVITY[i] for i in range(1, 7)]}")

    # 2. 特征提取
    print("\n[2/5] 提取特征 (每窗 8 频域 + 4 时域 / 通道 → 72 维)…")
    X_train_feat, feat_names = build_feature_matrix(X["train"])
    X_test_feat, _ = build_feature_matrix(X["test"])
    print(f"  训练特征: {X_train_feat.shape}, 测试特征: {X_test_feat.shape}")

    # 3. 训练决策树 (超参数搜索)
    print("\n[3/5] 训练决策树 (5-Fold GridSearchCV)…")
    clf = GridSearchCV(
        DecisionTreeClassifier(criterion="gini", random_state=42),
        {"max_depth": [6, 8, 10, 12, 15, None],
         "min_samples_split": [5, 10, 20, 50],
         "min_samples_leaf": [2, 5, 10, 20]},
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy", n_jobs=-1,
    )
    clf.fit(X_train_feat, y["train"])
    print(f"  最佳参数: {clf.best_params_}")
    print(f"  CV 准确率: {clf.best_score_ * 100:.2f}%")

    # 4. 评估
    print("\n[4/5] 评估…")
    y_pred = clf.predict(X_test_feat)
    train_acc = accuracy_score(y["train"], clf.predict(X_train_feat))
    test_acc = accuracy_score(y["test"], y_pred)
    print(f"  训练准确率: {train_acc * 100:.2f}%")
    print(f"  测试准确率: {test_acc * 100:.2f}%")

    # 消融实验：纯时域 vs 纯频域 vs 融合
    time_idx = [i for i, n in enumerate(feat_names)
                if n.split("_")[-1] in TIME_FEATURE_KEYS]
    freq_idx = [i for i, n in enumerate(feat_names)
                if n.split("_")[-1] in FREQ_FEATURE_KEYS]
    for label, idx_list in [("纯时域 (24维)", time_idx), ("纯频域 (48维)", freq_idx),
                             ("融合 (72维)", list(range(len(feat_names))))]:
        dt = DecisionTreeClassifier(criterion="gini", random_state=42, **clf.best_params_)
        dt.fit(X_train_feat[:, idx_list], y["train"])
        acc = accuracy_score(y["test"], dt.predict(X_test_feat[:, idx_list]))
        print(f"    {label} → {acc * 100:.2f}%")

    # 5. 生成可视化
    print("\n[5/5] 生成 7 张可视化图片…")
    plot_waveforms(X["train"], y["train"])
    plot_stft_band_energy(X["train"], y["train"])
    plot_fft_overlay(X["train"], y["train"])
    plot_band_energy(X["train"], y["train"])
    plot_spectral_scatter(X["train"], y["train"])
    plot_feature_importance(clf.best_estimator_.feature_importances_, feat_names)
    plot_confusion_matrix(y["test"], y_pred)

    for fname in sorted(os.listdir(SAVE_DIR)):
        print(f"  ✓ {fname}")

    print(f"\n  全部图片已保存至 {os.path.abspath(SAVE_DIR)}/")
    print("═" * 55)


if __name__ == "__main__":
    main()
