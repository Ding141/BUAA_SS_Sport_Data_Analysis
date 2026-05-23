"""
改进版: 时域峰值对齐 + 频域百分位包络
========================================
v1 问题: 动态动作各窗口步态相位不一致 → 直接平均抹平特征, std 过大
v2 方案:
  时域: 找到每个窗口的 heel-strike 主峰, 循环平移对齐后平均
  频域: 幅度谱本身相位不变, 改用 25th/75th 百分位替代 std
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FS = 50
N_SAMPLES = 128
DATA_DIR = "UCI HAR Dataset"
SAVE_DIR = "figures/analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "legend.fontsize": 8,
})

DYNAMIC = {1: "Walking", 2: "Walking Upstairs", 3: "Walking Downstairs"}
STATIC  = {4: "Sitting", 5: "Standing", 6: "Laying"}
COLORS = {
    1: "#2196F3", 2: "#4CAF50", 3: "#FF9800",
    4: "#F44336", 5: "#9C27B0", 6: "#00BCD4",
}


def load_data():
    path = os.path.join(DATA_DIR, "train", "Inertial Signals")
    channels = []
    for axis in ["x", "y", "z"]:
        for signal in ["body_acc"]:
            data = np.loadtxt(os.path.join(path, f"{signal}_{axis}_train.txt"), dtype=np.float32)
            channels.append(data)
    X = np.stack(channels, axis=-1)
    y = np.loadtxt(os.path.join(DATA_DIR, "train", "y_train.txt")).astype(int)
    return X, y


def fft_spectrum(signal):
    n = len(signal)
    vals = fft(signal)
    mag = np.abs(vals) / n
    mag = mag[: n // 2 + 1]
    mag[1:-1] *= 2
    freqs = fftfreq(n, 1 / FS)[: n // 2 + 1]
    return freqs, mag


def align_windows_to_peak(windows):
    """
    将每窗信号循环平移, 使主峰值对齐到窗口中心 (第64点)。

    原理: 找到每窗绝对值最大的位置 (heel-strike),
         循环平移使该峰落在 window_center, 再平均。
    对静态动作不做对齐 (无主峰, 对齐反而引入伪影)。
    """
    center = N_SAMPLES // 2
    aligned = np.zeros_like(windows)
    for i in range(len(windows)):
        sig = windows[i]
        # 找最大绝对值对应的采样点 (heel-strike 候选)
        peak_idx = np.argmax(np.abs(sig))
        shift = center - peak_idx
        aligned[i] = np.roll(sig, shift)
    return aligned


def compute_dynamic_time(windows, align=True):
    """动态动作时域: 对齐后求均值/std + 叠加3条原始样窗。"""
    if align:
        windows_aligned = align_windows_to_peak(windows)
    else:
        windows_aligned = windows

    mean_sig = windows_aligned.mean(axis=0)
    std_sig  = windows_aligned.std(axis=0)
    t = np.arange(N_SAMPLES) / FS

    # 取样窗: 在对齐后的窗中挑 3 条离均值最近的
    dists = np.sum((windows_aligned - mean_sig) ** 2, axis=1)
    sample_idx = np.argsort(dists)[:3]

    return t, mean_sig, std_sig, windows_aligned[sample_idx]


def compute_static_time(windows):
    """静态动作时域: 直接平均 (不晃, 无需对齐)。"""
    mean_sig = windows.mean(axis=0)
    std_sig  = windows.std(axis=0)
    t = np.arange(N_SAMPLES) / FS
    return t, mean_sig, std_sig


def compute_spectrum_percentiles(windows):
    """频域: 用 25/50/75 百分位替代 mean±std, 更稳健。"""
    all_mags = []
    for sig in windows:
        _, mag = fft_spectrum(sig)
        all_mags.append(mag)
    all_mags = np.array(all_mags)
    freqs, _ = fft_spectrum(windows[0])
    p25 = np.percentile(all_mags, 25, axis=0)
    p50 = np.percentile(all_mags, 50, axis=0)
    p75 = np.percentile(all_mags, 75, axis=0)
    return freqs, p50, p25, p75


def main():
    print("加载 UCI HAR 数据…")
    X, y = load_data()
    ch_idx = 0        # body_acc_x — 垂直轴, 步态特征最显著
    ch_name = "body_acc_x"

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # ═══════════════════════════════════════════════════════════
    # 左上: 动态动作 · 时域 (峰值对齐)
    # ═══════════════════════════════════════════════════════════
    ax = axes[0][0]
    for act_id, name in DYNAMIC.items():
        mask = y == act_id
        windows = X[mask, :, ch_idx]
        t, mean_sig, std_sig, samples = compute_dynamic_time(windows, align=True)

        c = COLORS[act_id]
        # 主曲线: 对齐后的均值
        ax.plot(t, mean_sig, color=c, linewidth=1.6, label=f"{name} (aligned mean)")
        # ±1σ 阴影
        ax.fill_between(t, mean_sig - std_sig, mean_sig + std_sig,
                        color=c, alpha=0.10)
        # 叠加 1 条最典型窗 (虚线)
        ax.plot(t, samples[0], color=c, linewidth=0.4, linestyle="--", alpha=0.6)

    ax.axvline(x=N_SAMPLES / 2 / FS, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.text(N_SAMPLES / 2 / FS + 0.02, ax.get_ylim()[1] * 0.95,
            "← peak aligned", fontsize=7, color="gray")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title(f"Time Domain — Dynamic (peak-aligned)\n({ch_name})")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(0, 2.56)
    ax.grid(True, alpha=0.3)

    # ═══════════════════════════════════════════════════════════
    # 右上: 动态动作 · 频域 (百分位包络)
    # ═══════════════════════════════════════════════════════════
    ax = axes[0][1]
    for act_id, name in DYNAMIC.items():
        mask = y == act_id
        windows = X[mask, :, ch_idx]
        freqs, p50, p25, p75 = compute_spectrum_percentiles(windows)

        mask_f = freqs <= 25
        c = COLORS[act_id]
        ax.plot(freqs[mask_f], p50[mask_f], color=c, linewidth=1.4,
                label=f"{name} (median)")
        ax.fill_between(freqs[mask_f], p25[mask_f], p75[mask_f],
                        color=c, alpha=0.12)
        # 主频标注
        peak_i = np.argmax(p50[mask_f])
        ax.annotate(f"{freqs[mask_f][peak_i]:.1f} Hz",
                    (freqs[mask_f][peak_i], p50[mask_f][peak_i]),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=7, color=c, ha="center")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Frequency Domain — Dynamic\n({ch_name})")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # ═══════════════════════════════════════════════════════════
    # 左下: 静态动作 · 时域 (无需对齐)
    # ═══════════════════════════════════════════════════════════
    ax = axes[1][0]
    for act_id, name in STATIC.items():
        mask = y == act_id
        windows = X[mask, :, ch_idx]
        t, mean_sig, std_sig = compute_static_time(windows)

        c = COLORS[act_id]
        ax.plot(t, mean_sig, color=c, linewidth=1.4, label=name)
        ax.fill_between(t, mean_sig - std_sig, mean_sig + std_sig,
                        color=c, alpha=0.12)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title(f"Time Domain — Static\n({ch_name})")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(0, 2.56)
    ax.grid(True, alpha=0.3)

    # ═══════════════════════════════════════════════════════════
    # 右下: 静态动作 · 频域
    # ═══════════════════════════════════════════════════════════
    ax = axes[1][1]
    for act_id, name in STATIC.items():
        mask = y == act_id
        windows = X[mask, :, ch_idx]
        freqs, p50, p25, p75 = compute_spectrum_percentiles(windows)

        mask_f = freqs <= 25
        c = COLORS[act_id]
        ax.plot(freqs[mask_f], p50[mask_f], color=c, linewidth=1.4,
                label=f"{name} (median)")
        ax.fill_between(freqs[mask_f], p25[mask_f], p75[mask_f],
                        color=c, alpha=0.12)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Frequency Domain — Static\n({ch_name})")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Improved: Dynamic vs Static — Time & Frequency Comparison\n"
        "v2: Peak-aligned time avg + median/percentile band + sample trace overlay",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    save_path = os.path.join(SAVE_DIR, "analysis_I_v2_aligned_comparison.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"已保存: {save_path}")


if __name__ == "__main__":
    main()
