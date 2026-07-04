"""
信号与系统大作业：基于频域分析的人体运动状态识别
====================================================
支持数据集: UCI HAR Dataset / WISDM Dataset
统一流水线: 数据加载 → 分窗 → FFT/STFT 频域特征提取 → 决策树分类 → 可视化
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import entropy as stats_entropy
from scipy.signal import spectrogram
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import glob
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

# ── 全局配置 ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = 50              # 默认采样频率 (UCI: 50 Hz, WISDM accel: 20 Hz)
N_SAMPLES = 128      # 每窗口采样点数
SAVE_DIR = os.path.join(PROJECT_ROOT, "figures", "分类流水线")
os.makedirs(SAVE_DIR, exist_ok=True)

# 说明: 该脚本实现基于频域（FFT/STFT）特征的人体动作识别流水线。
# 流程包括: 数据加载 → 分窗（若需） → 每窗口 FFT/STFT 特征提取 → 决策树训练与可视化。

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "font.sans-serif": ["Arial Unicode MS", "SimHei", "Heiti SC", "STHeiti"],
    "axes.unicode_minus": False,
})
sns.set_style("whitegrid")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   1. 数据集加载器 (多数据集支持)                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

class DataSet:
    """数据集抽象基类: load() → (X_raw, y, activity_names, fs, window_sec)"""

    def __init__(self, name, data_dir):
        self.name = name
        self.data_dir = data_dir

    def load(self):
        """返回 (X_raw, y, activity_dict, fs, n_samples_per_window)"""
        raise NotImplementedError


class UCIHARDataset(DataSet):
    """
    UCI HAR Dataset (预分窗).
    原始惯性信号: 50 Hz, 128 样本/窗口 (2.56 s), 6 通道 (body_acc xyz + body_gyro xyz).
    6 类: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying.
    """

    def __init__(self, data_dir=None):
        super().__init__("UCI HAR", data_dir or os.path.join(PROJECT_ROOT, "UCI HAR Dataset"))
        self.fs = 50
        self.n_samples = 128
        self.activities = {
            1: "Walking", 2: "Walking Upstairs", 3: "Walking Downstairs",
            4: "Sitting", 5: "Standing", 6: "Laying",
        }
        self.channel_names = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
        ]

    def load(self):
        # 返回格式说明:
        # X_raw[subset] -> ndarray, shape (n_windows, n_samples_per_window, n_channels)
        # y[subset]     -> ndarray, shape (n_windows,) 每个窗口对应的类别标签 (int)
        X_raw, y = {}, {}
        for subset in ["train", "test"]:
            path = os.path.join(self.data_dir, subset, "Inertial Signals")
            channels = []
            for axis in ["x", "y", "z"]:
                for signal in ["body_acc", "body_gyro"]:
                    data = np.loadtxt(os.path.join(path, f"{signal}_{axis}_{subset}.txt"))
                    channels.append(data)
            # channels 列表中元素均为 (n_windows, n_samples) 矩阵
            # np.stack(..., axis=-1) -> (n_windows, n_samples, n_channels=6)
            X_raw[subset] = np.stack(channels, axis=-1).astype(np.float32)
            y[subset] = np.loadtxt(
                os.path.join(self.data_dir, subset, f"y_{subset}.txt")
            ).astype(int)
        return X_raw, y, self.activities, self.channel_names, self.fs, self.n_samples


class WISDMDataset(DataSet):
    """
    WISDM Dataset (原始流式数据, 需要分窗).
    手机加速度计: 约 20 Hz.
    原始 18 类 → 默认映射为 6 类 (与 UCI 对齐的行走/慢跑/上楼/下楼/静坐/站立).
    """

    # 原始字母标签 → 数字标签 (仅保留有意义的 6 类)
    LETTER_MAP = {
        "A": 0,   # walking
        "B": 1,   # jogging
        "C": 2,   # stairs → 可视为 upstairs
        "D": 3,   # sitting
        "E": 4,   # standing
    }
    ACTIVITIES = {
        0: "Walking", 1: "Jogging", 2: "Stairs",
        3: "Sitting", 4: "Standing",
    }
    N_CLASSES = 5
    FS = 20  # WISDM 加速度计实际采样率为 20 Hz
    WINDOW_SIZE = 128     # 与 UCI 一致的窗口长度
    WINDOW_STRIDE = 64    # 50% 重叠

    def __init__(self, data_dir=None):
        super().__init__("WISDM", data_dir or os.path.join(PROJECT_ROOT, "wisdm-dataset"))
        self.channel_names = ["accel_x", "accel_y", "accel_z"]

    def _load_raw_files(self):
        """加载所有 phone/accel 文件，返回合并后的 (timestamps, x, y, z, label_id) 列表。"""
        accel_dir = os.path.join(self.data_dir, "raw", "phone", "accel")
        files = sorted(glob.glob(os.path.join(accel_dir, "data_*_accel_phone.txt")))

        all_data = []
        for fpath in files:
            with open(fpath, "r") as f:
                for line in f:
                    line = line.strip().rstrip(";")
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    letter = parts[1].strip().upper()
                    if letter not in self.LETTER_MAP:
                        continue
                    try:
                        ts = int(parts[2].strip())
                        x = float(parts[3].strip())
                        y_val = float(parts[4].strip())
                        z = float(parts[5].strip())
                    except (ValueError, IndexError):
                        continue
                    label = self.LETTER_MAP[letter]
                    # 记录为五元组: (timestamp, x, y, z, label_id)
                    all_data.append((ts, x, y_val, z, label))

        return sorted(all_data, key=lambda r: r[0])

    def _segment_windows(self, records):
        """对排序后的记录做滑动窗，返回 (n_windows, 128, 3) 和标签。"""
        arr = np.array(records)  # columns: ts, x, y, z, label
        X_windows, y_windows = [], []

        i = 0
        while i + self.WINDOW_SIZE <= len(arr):
            window = arr[i: i + self.WINDOW_SIZE]
            # 只保留窗口内所有样本属于同一类的窗口
            labels_in_window = window[:, 4]
            if np.all(labels_in_window == labels_in_window[0]):
                accel_data = window[:, 1:4].astype(np.float32)  # (128, 3)
                X_windows.append(accel_data)
                y_windows.append(int(labels_in_window[0]))
            i += self.WINDOW_STRIDE

        # 返回: X_windows (n_windows, window_size, 3), y_windows (n_windows,)
        return np.stack(X_windows), np.array(y_windows)

    def load(self):
        print("  [WISDM] 加载手机加速度计原始数据…")
        records = self._load_raw_files()
        print(f"  [WISDM] 共 {len(records):,} 条有效记录 (过滤后 5 类)")

        # 按时间排序后，前 80% 训练，后 20% 测试 (按用户分割更合理)
        print(f"  [WISDM] 滑窗分窗 (窗长={self.WINDOW_SIZE}, 步长={self.WINDOW_STRIDE})…")
        X_all, y_all = self._segment_windows(records)
        print(f"  [WISDM] 总窗口数: {X_all.shape[0]}")

        # 时间序划分 (不打乱时间顺序), 80/20 split
        split_idx = int(0.8 * len(X_all))
        X_raw = {
            "train": X_all[:split_idx],
            "test": X_all[split_idx:],
        }
        y = {
            "train": y_all[:split_idx],
            "test": y_all[split_idx:],
        }

        # 扩展为 6 通道以复用 UCI 的特征提取 (后 3 通道填 0 表示无陀螺仪)
        for k in ["train", "test"]:
            padding = np.zeros((X_raw[k].shape[0], self.WINDOW_SIZE, 3), dtype=np.float32)
            X_raw[k] = np.concatenate([X_raw[k], padding], axis=-1)

        self.channel_names = ["accel_x", "accel_y", "accel_z",
                              "gyro_x(na)", "gyro_y(na)", "gyro_z(na)"]
        return X_raw, y, self.ACTIVITIES, self.channel_names, self.FS, self.WINDOW_SIZE


def get_dataset(name):
    """工厂函数: 根据名称返回数据集实例。"""
    name_lower = name.lower()
    if name_lower in ("uci", "ucihar", "uci_har"):
        return UCIHARDataset()
    elif name_lower in ("wisdm",):
        return WISDMDataset()
    else:
        raise ValueError(f"未知数据集: {name}。支持: uci / wisdm")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    2. 频域分析 & 特征提取                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def compute_fft_spectrum(window_signal, fs):
    """单窗口单通道 → FFT 单边幅度谱 + 频率轴。"""
    n = len(window_signal)
    fft_vals = fft(window_signal)
    # 归一化为幅度谱: |FFT|/N，取单边频谱并把中间分量乘 2（直流与 Nyquist 不变）
    mag = np.abs(fft_vals) / n
    mag = mag[: n // 2 + 1]
    mag[1:-1] *= 2
    freqs = fftfreq(n, 1 / fs)[: n // 2 + 1]
    # 返回频率轴与对应幅度（单边）
    return freqs, mag


def extract_frequency_features(mag, freqs):
    """从单边幅度谱提取 8 个频域特征 (dict)。"""
    f = {}
    total_mag = np.sum(mag)
    eps = 1e-12

    f["PeakFreq"] = freqs[np.argmax(mag)]

    if total_mag > eps:
        f["MeanFreq"] = np.sum(freqs * mag) / total_mag
        cumulative = np.cumsum(mag)
        f["MedianFreq"] = freqs[np.searchsorted(cumulative, total_mag / 2)]
        f["SpectralEnergy"] = np.sum(mag ** 2)
        prob = mag / total_mag
        f["SpectralEntropy"] = stats_entropy(prob + eps)
        # 频带能量 (归一化到 Nyquist)
        nyq = freqs[-1]
        f["BandEnergy_Low"] = np.sum(mag[(freqs >= 0) & (freqs < nyq * 0.2)] ** 2)
        f["BandEnergy_Mid"] = np.sum(mag[(freqs >= nyq * 0.2) & (freqs < nyq * 0.6)] ** 2)
        f["BandEnergy_High"] = np.sum(mag[(freqs >= nyq * 0.6)] ** 2)
    else:
        for k in ("MeanFreq", "MedianFreq", "SpectralEnergy", "SpectralEntropy",
                  "BandEnergy_Low", "BandEnergy_Mid", "BandEnergy_High"):
            f[k] = 0.0
    # 说明: 频域特征含义
    # - PeakFreq: 幅度谱峰值对应的频率
    # - MeanFreq/MedianFreq: 频率统计量, 描述能量分布中心
    # - SpectralEnergy: 幅度平方和, 代表带宽内总体能量
    # - SpectralEntropy: 频谱的熵, 描述频谱复杂度
    # - BandEnergy_*: 分别计算低/中/高频带的能量用于区分活动
    return f


def extract_time_features(signal):
    """提取 4 个时域特征。"""
    return {
        "Mean": np.mean(signal),
        "Var": np.var(signal),
        "PeakToPeak": np.ptp(signal),
        "ZeroCrossingRate": np.sum(np.diff(np.signbit(signal))) / len(signal),
    }


def build_feature_matrix(raw_data, channel_names, fs, verbose=True):
    """
    逐窗口逐通道提取频域+时域特征。
    返回 X (n_windows, n_features), feature_names (list).
    """
    n_windows = raw_data.shape[0]
    n_channels = raw_data.shape[2]
    all_rows = []

    for i in range(n_windows):
        row = []
        for ch in range(n_channels):
            signal = raw_data[i, :, ch]
            freqs, mag = compute_fft_spectrum(signal, fs)
            freq_f = extract_frequency_features(mag, freqs)
            time_f = extract_time_features(signal)
            row.extend(freq_f.values())
            row.extend(time_f.values())
        all_rows.append(row)

        if verbose and (i + 1) % max(1, n_windows // 4) == 0:
            print(f"  处理进度: {i + 1}/{n_windows} 窗口 ({(i + 1) / n_windows * 100:.0f}%)")

    # 生成列名
    col_names = []
    # 通过调用特征提取函数获取特征键名（顺序需与上文 row.extend 保持一致）
    # 注意: 这里传入空数组只为获取 key 列表，函数内部有防护逻辑以避免除零异常。
    freq_tags = list(extract_frequency_features(np.zeros(2), np.zeros(2)).keys())
    time_tags = list(extract_time_features(np.zeros(2)).keys())
    for ch in channel_names:
        for tag in freq_tags:
            col_names.append(f"{ch}_{tag}")
        for tag in time_tags:
            col_names.append(f"{ch}_{tag}")

    return np.array(all_rows), col_names


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      3. 可视化函数                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

def plot_waveforms(raw_data, labels, activity_names, n_samples, fs, dataset_name):
    """图1: 各类动作的典型波形。"""
    n_classes = len(activity_names)
    ncols = min(3, n_classes)
    nrows = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             squeeze=False)
    t = np.arange(n_samples) / fs
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for i, (act_id, act_name) in enumerate(sorted(activity_names.items())):
        ax = axes[i // ncols][i % ncols]
        idxs = np.where(labels == act_id)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[len(idxs) // 2]
        for ch_idx in range(min(3, raw_data.shape[2])):
            ax.plot(t, raw_data[idx, :, ch_idx], linewidth=0.7,
                    color=colors[ch_idx], alpha=0.85)
        ax.set_title(act_name, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Accel (g)")
        ax.set_ylim(-5, 15)

    # 注意: y 轴范围设置为 [-5, 15] 以兼容 WISDM（有时单位/量纲不同），可根据实际数据调整。

    # 隐藏空子图
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"[{dataset_name}] Figure 1: Raw Acceleration Waveforms",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "figure1_waveforms.png"))
    plt.close(fig)


def plot_stft_spectrogram(raw_data, labels, activity_names, fs, dataset_name):
    """图2: 几类典型动作的 STFT 时频谱。"""
    sample_acts = list(sorted(activity_names.keys()))[:4]
    fig, axes = plt.subplots(len(sample_acts), 1, figsize=(12, 2.8 * len(sample_acts)),
                             squeeze=False)

    for row, act_id in enumerate(sample_acts):
        idxs = np.where(labels == act_id)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[len(idxs) // 2]
        f_vals, t_vals, Sxx = spectrogram(
            raw_data[idx, :, 0], fs=fs, nperseg=min(32, raw_data.shape[1] // 4),
            noverlap=min(24, raw_data.shape[1] // 5), nfft=min(64, raw_data.shape[1] // 2),
        )
        # Sxx 是功率谱密度，使用 10*log10 显示为 dB 以增强时频对比度
        ax = axes[row][0]
        im = ax.pcolormesh(t_vals, f_vals, 10 * np.log10(Sxx + 1e-12),
                           shading="gouraud", cmap="viridis")
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(f"{activity_names[act_id]}", fontsize=10)
        plt.colorbar(im, ax=ax, label="dB")

    axes[-1][0].set_xlabel("Time (s)")
    fig.suptitle(f"[{dataset_name}] Figure 2: STFT Spectrograms",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "figure2_stft.png"))
    plt.close(fig)


def plot_spectrum_comparison(raw_data, labels, activity_names, fs, dataset_name):
    """图3: 不同动作的 FFT 幅度谱对比。"""
    act_list = sorted(activity_names.keys())
    if len(act_list) < 2:
        return
    pairs = [(act_list[0], act_list[1]),
             (act_list[0], act_list[-1]) if len(act_list) > 2 else (act_list[0], act_list[1]),
             (act_list[len(act_list) // 2], act_list[-1])]
    pairs = list(dict.fromkeys(pairs))  # 去重

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), squeeze=False)

    for j, (a1, a2) in enumerate(pairs):
        ax = axes[0][j]
        for act_id, color, ls in [(a1, "#2196F3", "-"), (a2, "#FF5722", "--")]:
            idxs = np.where(labels == act_id)[0]
            if len(idxs) == 0:
                continue
            idx = idxs[len(idxs) // 2]
            freqs, mag = compute_fft_spectrum(raw_data[idx, :, 0], fs)
            mask = freqs <= fs / 2 * 0.8
            ax.plot(freqs[mask], mag[mask], color=color, linestyle=ls, linewidth=1.0,
                    label=activity_names[act_id])
            peak_i = np.argmax(mag[mask])
            ax.annotate(f"{freqs[mask][peak_i]:.1f} Hz",
                        (freqs[mask][peak_i], mag[mask][peak_i]),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=7, color=color, ha="center")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"{activity_names[a1]} vs {activity_names[a2]}", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(f"[{dataset_name}] Figure 3: FFT Magnitude Spectrum Comparison (accel_x)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "figure3_spectrum.png"))
    plt.close(fig)


def plot_feature_scatter(X, y, feature_names, activity_names, dataset_name):
    """图4: 频域特征散点图矩阵。"""
    # 选最有区分度的几对特征
    candidates = [n for n in feature_names if any(
        kw in n for kw in ["MeanFreq", "SpectralEntropy", "SpectralEnergy", "PeakFreq"])]
    pairs = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if len(pairs) < 6:
                pairs.append((candidates[i], candidates[j]))

    if not pairs or len(pairs) < 2:
        pairs = [(feature_names[0], feature_names[1]),
                 (feature_names[2], feature_names[3])]

    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(activity_names))))

    for k, (fx, fy) in enumerate(pairs):
        ax = axes[k // ncols][k % ncols]
        ix, iy = feature_names.index(fx), feature_names.index(fy)
        for act_id in sorted(activity_names.keys()):
            mask = y == act_id
            ax.scatter(X[mask, ix], X[mask, iy], s=3, alpha=0.4,
                       color=colors[act_id % 10], label=activity_names[act_id],
                       rasterized=True)
        ax.set_xlabel(fx, fontsize=7)
        ax.set_ylabel(fy, fontsize=7)
        ax.tick_params(labelsize=6)

    # 隐藏空子图
    for j in range(k + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    handles, labels_ax = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_ax, loc="lower center", ncol=min(6, len(activity_names)),
               fontsize=7, markerscale=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"[{dataset_name}] Figure 4: Frequency Feature Distributions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(os.path.join(SAVE_DIR, "figure4_scatter.png"))
    plt.close(fig)


def plot_decision_tree(clf, feature_names, activity_names, dataset_name):
    """图5: 决策树结构图。"""
    fig, ax = plt.subplots(figsize=(22, 11))
    plot_tree(
        clf, max_depth=3, feature_names=feature_names,
        class_names=[activity_names[i] for i in sorted(activity_names.keys())],
        filled=True, rounded=True, fontsize=7, ax=ax, impurity=False,
        proportion=True,
    )
    ax.set_title(f"[{dataset_name}] Figure 5: Decision Tree — Top 3 Levels",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "figure5_tree.png"))
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, activity_names, dataset_name, tag=""):
    """图6: 混淆矩阵。"""
    cm = confusion_matrix(y_true, y_pred)
    labels_list = [activity_names[i] for i in sorted(activity_names.keys())]
    fig, ax = plt.subplots(figsize=(max(6, len(labels_list) * 1.1),
                                    max(5, len(labels_list) * 0.9)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels_list, yticklabels=labels_list,
                linewidths=0.5, annot_kws={"fontsize": 10})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    suffix = f" ({tag})" if tag else ""
    ax.set_title(f"[{dataset_name}] Figure 6: Confusion Matrix{suffix}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = f"figure6_cm{'_' + tag if tag else ''}.png"
    fig.savefig(os.path.join(SAVE_DIR, fname))
    plt.close(fig)


def plot_feature_importance(importances, feature_names, dataset_name):
    """图7: 特征重要性 Top-20。"""
    top_n = min(20, len(feature_names))
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    colors = ["#FF5722" if ("Freq" in feature_names[i] or "Band" in feature_names[i]
                            or "Spectral" in feature_names[i] or "Entropy" in feature_names[i])
              else "#2196F3" for i in sorted_idx]
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(range(top_n), importances[sorted_idx][::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx][::-1], fontsize=8)
    ax.set_xlabel("Gini Importance")
    ax.set_title(f"[{dataset_name}] Figure 7: Feature Importances (Red=Freq, Blue=Time)",
                 fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#FF5722", label="Frequency Domain"),
                        Patch(color="#2196F3", label="Time Domain")],
              fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "figure7_importance.png"))
    plt.close(fig)


def plot_band_energy(raw_data, labels, activity_names, fs, dataset_name):
    """图8: 频带能量分布饼图。"""
    sample_acts = sorted(activity_names.keys())[:4]
    bands = {"Low (0–20%)": (0, 0.2), "Mid (20–60%)": (0.2, 0.6), "High (60%–Nyq)": (0.6, 1.0)}
    band_colors = ["#81C784", "#64B5F6", "#FF8A65"]
    nyq = fs / 2

    fig, axes = plt.subplots(1, len(sample_acts), figsize=(4 * len(sample_acts), 4))
    if len(sample_acts) == 1:
        axes = [axes]

    for ax, act_id in zip(axes, sample_acts):
        idxs = np.where(labels == act_id)[0][:100]
        if len(idxs) == 0:
            continue
        avg_energies = []
        for _, (lo_pct, hi_pct) in bands.items():
            e_sum = 0
            for idx in idxs:
                freqs, mag = compute_fft_spectrum(raw_data[idx, :, 0], fs)
                mask = (freqs >= nyq * lo_pct) & (freqs < nyq * hi_pct)
                e_sum += np.sum(mag[mask] ** 2)
            avg_energies.append(e_sum / len(idxs))
        total = sum(avg_energies)
        pcts = [e / total * 100 if total > 0 else 0 for e in avg_energies]
        ax.pie(pcts, labels=list(bands.keys()), colors=band_colors,
               autopct="%1.1f%%", startangle=90, explode=(0, 0.05, 0.1),
               textprops={"fontsize": 8})
        ax.set_title(activity_names[act_id], fontsize=10)

    fig.suptitle(f"[{dataset_name}] Figure 8: Frequency Band Energy Distribution (accel_x)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "figure8_band.png"))
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       4. 主流程                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def print_section(title):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def run_pipeline(dataset):
    """对给定数据集执行完整的特征提取+建模+评估+可视化流水线。"""
    global SAVE_DIR
    ds_name = dataset.name
    ds_name_map = {"UCI HAR": "uci", "WISDM": "wisdm"}
    ds_tag = ds_name_map.get(ds_name, ds_name.lower().replace(" ", "_"))
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 1. 加载数据 ──
    print_section(f"[{ds_name}] 1/5 数据加载")
    X_raw, y, activity_names, channel_names, fs, n_samples = dataset.load()
    # X_raw: dict with 'train'/'test' arrays; 每个 array 形状如 (n_windows, n_samples, n_channels)
    print(f"  训练窗口: {X_raw['train'].shape[0]}, 测试窗口: {X_raw['test'].shape[0]}")
    print(f"  采样率: {fs} Hz, 窗口: {n_samples} 点 ({n_samples / fs:.2f} s)")
    print(f"  通道数: {X_raw['train'].shape[2]} ({', '.join(channel_names[:3])}...)")
    print(f"  类别数: {len(activity_names)} → {list(activity_names.values())}")
    for sub in ["train", "test"]:
        cnt = Counter(y[sub])
        print(f"  {sub}: {dict(sorted(cnt.items()))}")

    n_features_per_ch = 12  # 8 freq + 4 time
    print(f"  特征维度: {X_raw['train'].shape[2]} 通道 × {n_features_per_ch} = "
          f"{X_raw['train'].shape[2] * n_features_per_ch} 维")

    # ── 2. 特征提取 ──
    print_section(f"[{ds_name}] 2/5 特征提取 (FFT 频域 8 + 时域 4 / 通道)")
    X_train, feat_names = build_feature_matrix(X_raw["train"], channel_names, fs)
    X_test, _ = build_feature_matrix(X_raw["test"], channel_names, fs)
    # 注意: build_feature_matrix 返回的 X_train/X_test 已经是每个窗口的一维特征向量
    print(f"  训练特征: {X_train.shape}, 测试特征: {X_test.shape}")

    # ── 3. 超参数搜索 + 训练 ──
    print_section(f"[{ds_name}] 3/5 决策树超参数搜索 (5-Fold CV)")
    param_grid = {
        "max_depth": [6, 8, 10, 12, 15, None],
        "min_samples_split": [5, 10, 20, 50],
        "min_samples_leaf": [2, 5, 10, 20],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(criterion="gini", random_state=42),
        param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy", n_jobs=-1,
    )
    grid.fit(X_train, y["train"])
    print(f"  最佳参数: {grid.best_params_}")
    print(f"  CV 准确率: {grid.best_score_ * 100:.2f}%")

    clf = DecisionTreeClassifier(criterion="gini", random_state=42, **grid.best_params_)
    clf.fit(X_train, y["train"])
    y_pred = clf.predict(X_test)
    train_acc = accuracy_score(y["train"], clf.predict(X_train))
    test_acc = accuracy_score(y["test"], y_pred)
    print(f"  训练准确率: {train_acc * 100:.2f}%")
    print(f"  测试准确率: {test_acc * 100:.2f}%")
    print(f"  树深度: {clf.get_depth()}, 叶节点: {clf.get_n_leaves()}")

    # ── 4. 评估 ──
    print_section(f"[{ds_name}] 4/5 分类评估")
    print(f"\n  准确率:      {test_acc * 100:.2f}%")
    print(f"  宏平均 F1:   {f1_score(y['test'], y_pred, average='macro') * 100:.2f}%")
    print(f"  加权平均 F1: {f1_score(y['test'], y_pred, average='weighted') * 100:.2f}%")
    report = classification_report(
        y["test"], y_pred,
        target_names=[activity_names[i] for i in sorted(activity_names.keys())],
        digits=4,
    )
    print(f"\n{report}")

    # 特征重要性
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1]
    print(f"  Top-15 特征重要性:")
    for rank, idx in enumerate(top_idx[:15], 1):
        tag = "[频域]" if any(kw in feat_names[idx] for kw in
            ["Freq", "Spectral", "Band", "Entropy"]) else "[时域]"
        print(f"  {rank:2d}. {tag} {feat_names[idx]:45s} {importances[idx]:.4f}")

    # 消融实验
    time_idx = [i for i, n in enumerate(feat_names)
                if any(n.endswith(t) for t in ["Mean", "Var", "PeakToPeak", "ZeroCrossingRate"])]
    freq_idx = [i for i, n in enumerate(feat_names)
                if any(kw in n for kw in ["Freq", "Spectral", "Band", "Entropy"])]

    print(f"\n  消融实验:")
    results = {}
    for label, idx_list in [("纯时域", time_idx), ("纯频域", freq_idx), ("融合", list(range(len(feat_names))))]:
        c = DecisionTreeClassifier(criterion="gini", random_state=42, **grid.best_params_)
        c.fit(X_train[:, idx_list], y["train"])
        acc = accuracy_score(y["test"], c.predict(X_test[:, idx_list]))
        results[label] = acc
        print(f"    {label} ({len(idx_list)}维) → {acc * 100:.2f}%")
    delta = results["融合"] - results["纯时域"]
    print(f"    频域特征贡献: {delta * 100:+.2f} 个百分点")

    # ── 5. 可视化 ──
    print_section(f"[{ds_name}] 5/5 生成可视化")
    plot_waveforms(X_raw["train"], y["train"], activity_names, n_samples, fs, ds_name)
    print(f"  ✓ {ds_tag}/figure1_waveforms.png")

    plot_stft_spectrogram(X_raw["train"], y["train"], activity_names, fs, ds_name)
    print(f"  ✓ {ds_tag}/figure2_stft.png")

    plot_spectrum_comparison(X_raw["train"], y["train"], activity_names, fs, ds_name)
    print(f"  ✓ {ds_tag}/figure3_spectrum.png")

    plot_feature_scatter(X_train, y["train"], feat_names, activity_names, ds_name)
    print(f"  ✓ {ds_tag}/figure4_scatter.png")

    plot_decision_tree(clf, feat_names, activity_names, ds_name)
    print(f"  ✓ {ds_tag}/figure5_tree.png")

    plot_confusion_matrix(y["test"], y_pred, activity_names, ds_name)
    print(f"  ✓ {ds_tag}/figure6_cm.png")

    plot_feature_importance(importances, feat_names, ds_name)
    print(f"  ✓ {ds_tag}/figure7_importance.png")

    plot_band_energy(X_raw["train"], y["train"], activity_names, fs, ds_name)
    print(f"  ✓ {ds_tag}/figure8_band.png")

    return test_acc, results


# ── 入口 ──────────────────────────────────────────────────────────────

def main():
    print("╔" + "═" * 58 + "╗")
    print("║  信号与系统大作业：运动传感器频域分析与分类                  ║")
    print("║  北京航空航天大学 · 2026 春季学期                            ║")
    print("║  支持: UCI HAR Dataset / WISDM Dataset                      ║")
    print("╚" + "═" * 58 + "╝")

    # 选择数据集
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if dataset_arg.lower() == "all":
        dataset_names = ["uci", "wisdm"]
    else:
        dataset_names = [dataset_arg]

    all_results = {}
    for ds_name in dataset_names:
        print(f"\n{'█' * 60}")
        print(f"█  数据集: {ds_name.upper()}")
        print(f"{'█' * 60}")
        dataset = get_dataset(ds_name)
        acc, ablation = run_pipeline(dataset)
        all_results[ds_name] = {"accuracy": acc, "ablation": ablation}

    # 汇总对比
    print(f"\n{'═' * 60}")
    print(f"  📊 汇总对比")
    print(f"{'═' * 60}")
    for ds_name, res in all_results.items():
        print(f"\n  [{ds_name.upper()}]")
        print(f"    测试准确率:   {res['accuracy'] * 100:.2f}%")
        for k, v in res["ablation"].items():
            print(f"    {k}:         {v * 100:.2f}%")
        delta = res["ablation"]["融合"] - res["ablation"]["纯时域"]
        print(f"    频域提升:     {delta * 100:+.2f} 个百分点")

    print(f"\n  所有图像已保存至: {os.path.abspath('figures')}/")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
