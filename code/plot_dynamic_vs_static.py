"""
UCI HAR 时域+频域对比图：动态动作3类 vs 静态动作3类
=====================================================
2×2 子图布局:
  [动态 时域] [动态 频域]
  [静态 时域] [静态 频域]
每个子图内绘制该类所有窗口的平均 ± 标准差阴影。
"""

import numpy as np
from scipy.fft import fft, fftfreq
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
    X = np.stack(channels, axis=-1)  # (7352, 128, 3): body_acc x/y/z
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


def compute_mean_and_std(X, y, act_id, ch_idx, domain="time"):
    """计算某活动某通道的平均曲线 ± 标准差。"""
    mask = y == act_id
    windows = X[mask, :, ch_idx]

    if domain == "time":
        mean = windows.mean(axis=0)
        std  = windows.std(axis=0)
        t = np.arange(N_SAMPLES) / FS
        return t, mean, std
    else:
        all_mags = []
        for i in range(len(windows)):
            _, mag = fft_spectrum(windows[i])
            all_mags.append(mag)
        all_mags = np.array(all_mags)
        freqs, _ = fft_spectrum(windows[0])
        return freqs, all_mags.mean(axis=0), all_mags.std(axis=0)


def main():
    print("加载 UCI HAR 数据…")
    X, y = load_data()
    ch_idx = 0        # body_acc_x — 垂直/上下方向 (重力轴), 步态特征最显著
    ch_name = "body_acc_x (vertical)"

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # ── 左上: 动态动作 · 时域 ──
    ax = axes[0][0]
    for act_id, name in DYNAMIC.items():
        t, mean_sig, std_sig = compute_mean_and_std(X, y, act_id, ch_idx, "time")
        c = COLORS[act_id]
        ax.plot(t, mean_sig, color=c, linewidth=1.2, label=name)
        ax.fill_between(t, mean_sig - std_sig, mean_sig + std_sig,
                        color=c, alpha=0.12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title(f"Time Domain — Dynamic Activities\n({ch_name})")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 2.56)
    ax.grid(True, alpha=0.3)

    # ── 右上: 动态动作 · 频域 ──
    ax = axes[0][1]
    for act_id, name in DYNAMIC.items():
        freqs, mean_mag, std_mag = compute_mean_and_std(X, y, act_id, ch_idx, "freq")
        mask = freqs <= 25
        c = COLORS[act_id]
        ax.plot(freqs[mask], mean_mag[mask], color=c, linewidth=1.2, label=name)
        ax.fill_between(freqs[mask],
                        np.maximum(0, mean_mag[mask] - std_mag[mask]),
                        mean_mag[mask] + std_mag[mask],
                        color=c, alpha=0.12)
        # 标注主峰
        peak_i = np.argmax(mean_mag[mask])
        ax.annotate(f"{freqs[mask][peak_i]:.1f} Hz",
                    (freqs[mask][peak_i], mean_mag[mask][peak_i]),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=7, color=c, ha="center")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Frequency Domain — Dynamic Activities\n({ch_name})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── 左下: 静态动作 · 时域 ──
    ax = axes[1][0]
    for act_id, name in STATIC.items():
        t, mean_sig, std_sig = compute_mean_and_std(X, y, act_id, ch_idx, "time")
        c = COLORS[act_id]
        ax.plot(t, mean_sig, color=c, linewidth=1.2, label=name)
        ax.fill_between(t, mean_sig - std_sig, mean_sig + std_sig,
                        color=c, alpha=0.12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title(f"Time Domain — Static Activities\n({ch_name})")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 2.56)
    ax.grid(True, alpha=0.3)

    # ── 右下: 静态动作 · 频域 ──
    ax = axes[1][1]
    for act_id, name in STATIC.items():
        freqs, mean_mag, std_mag = compute_mean_and_std(X, y, act_id, ch_idx, "freq")
        mask = freqs <= 25
        c = COLORS[act_id]
        ax.plot(freqs[mask], mean_mag[mask], color=c, linewidth=1.2, label=name)
        ax.fill_between(freqs[mask],
                        np.maximum(0, mean_mag[mask] - std_mag[mask]),
                        mean_mag[mask] + std_mag[mask],
                        color=c, alpha=0.12)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Frequency Domain — Static Activities\n({ch_name})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 总标题
    fig.suptitle(
        "Dynamic vs Static Activities: Time & Frequency Domain Comparison\n"
        "All curves are mean over training windows (±1σ shaded)",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    save_path = os.path.join(SAVE_DIR, "analysis_I_dynamic_vs_static_2x2.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"已保存: {save_path}")

    # ── 终端补充统计 ──
    print("\n─ 频域关键指标对比 ─")
    print(f"  {'Activity':<22s} {'Peak(Hz)':<10s} {'MeanF(Hz)':<10s} {'Energy':<12s} {'Entropy':<10s}")
    print("  " + "─" * 65)
    for act_id in range(1, 7):
        freqs, mean_mag, _ = compute_mean_and_std(X, y, act_id, ch_idx, "freq")
        mask = freqs <= 25
        mag = mean_mag[mask]; f = freqs[mask]
        total = np.sum(mag)
        peak = f[np.argmax(mag)]
        meanf = np.sum(f * mag) / (total + 1e-12)
        energy = np.sum(mag ** 2)
        entropy = -np.sum((mag / (total + 1e-12)) * np.log(mag / (total + 1e-12) + 1e-12))
        name = {**DYNAMIC, **STATIC}[act_id]
        tag = "[动]" if act_id <= 3 else "[静]"
        print(f"  {tag} {name:<19s} {peak:>8.2f}  {meanf:>8.2f}  {energy:>10.4f}  {entropy:>8.4f}")


if __name__ == "__main__":
    main()
