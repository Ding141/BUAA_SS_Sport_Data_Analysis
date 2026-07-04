"""
改进版 STFT 对比图: 用频带能量时间演进折线图替代 2D 热力图
============================================================
目的: 让不同动作在时频域的差异可以精确读出, 而非仅靠颜色深浅判断

输出:
  figures/频谱与特征分析/12_STFT频带能量时间演进.png
  figures/频谱与特征分析/13_频谱时间切片对比.png
"""

import numpy as np
from scipy.signal import spectrogram
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings("ignore")

FS = 50
DATA_DIR = "UCI HAR Dataset"
SAVE_DIR = os.path.join("figures", "频谱与特征分析")
os.makedirs(SAVE_DIR, exist_ok=True)

活动名称 = {1: "行走", 4: "静坐", 5: "站立", 6: "躺卧"}
活动颜色 = {1: "#2196F3", 4: "#F44336", 5: "#9C27B0", 6: "#00BCD4"}
频带颜色 = {
    "DC–2 Hz":   "#1B5E20",
    "2–5 Hz":    "#4CAF50",
    "5–10 Hz":   "#FFC107",
    "10–25 Hz":  "#F44336",
}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "legend.fontsize": 8,
    "font.sans-serif": ["Arial Unicode MS", "SimHei", "Heiti SC", "STHeiti"],
    "axes.unicode_minus": False,
})


def load_data():
    path = os.path.join(DATA_DIR, "train", "Inertial Signals")
    channels = []
    for axis in ["x", "y", "z"]:
        for signal in ["body_acc", "body_gyro"]:
            data = np.loadtxt(os.path.join(path, f"{signal}_{axis}_train.txt"), dtype=np.float32)
            channels.append(data)
    X = np.stack(channels, axis=-1)
    y = np.loadtxt(os.path.join(DATA_DIR, "train", "y_train.txt")).astype(int)
    return X, y


def figure_band_energy_evolution(X, y):
    """
    图 A: 频带能量时间演进折线图
    -----
    对行走和静坐各取一个代表性窗口, 将 STFT 的频谱按频带聚合为能量,
    在时间轴上画折线。折线图可以直接读出数值, 对比行走 vs 静坐的差异。
    """
    act_pairs = [(1, 4), (1, 6)]  # 行走vs静坐, 行走vs躺卧
    channels = [(0, "acc_x"), (2, "acc_z")]
    bands = [
        ("DC–2 Hz",   0, 2),
        ("2–5 Hz",    2, 5),
        ("5–10 Hz",   5, 10),
        ("10–25 Hz", 10, 25),
    ]
    nperseg, noverlap, nfft = 32, 24, 64

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    for row, (act_a, act_b) in enumerate(act_pairs):
        for col, (ch_idx, ch_name) in enumerate(channels):
            ax = fig.add_subplot(gs[row, col])

            for act_id, ls, lw in [(act_a, "-", 1.8), (act_b, "--", 1.5)]:
                idxs = np.where(y == act_id)[0]
                win = X[idxs[len(idxs) // 2], :, ch_idx]

                f_vals, t_vals, Sxx = spectrogram(win, fs=FS,
                                                  nperseg=nperseg, noverlap=noverlap, nfft=nfft)

                for band_name, lo, hi in bands:
                    mask = (f_vals >= lo) & (f_vals < hi)
                    band_energy = np.sum(Sxx[mask, :], axis=0)
                    ax.plot(t_vals, band_energy,
                            color=频带颜色[band_name],
                            linestyle=ls, linewidth=lw, alpha=0.9,
                            label=f"{活动名称[act_id]} {band_name}" if col == 0 else "")

            ax.set_title(f"{活动名称[act_a]} vs {活动名称[act_b]} — {ch_name}", fontsize=11)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("频带能量")
            ax.set_xlim(t_vals[0], t_vals[-1])

    # 图例放在底部
    handles, labels = ax.get_legend_handles_labels()
    legend_ax = fig.add_axes([0.12, 0.02, 0.76, 0.04])
    legend_ax.set_axis_off()
    legend_ax.legend(handles, labels, loc="center", ncol=4, fontsize=7.5,
                     title="实线=行走  虚线=静态", title_fontsize=8)

    fig.suptitle("STFT 频带能量时间演进 — 折线图替代热力图", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(os.path.join(SAVE_DIR, "12_STFT频带能量时间演进.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 12_STFT频带能量时间演进.png")


def figure_spectral_slices(X, y):
    """
    图 B: 频谱时间切片对比
    -----
    在同一窗口内的 3 个时间点各取一次 FFT 频谱切片,
    将行走和静坐的切片分别叠加, 展示频谱随时间的一致性。
    折线可以直接对比频率轴上的幅度差异。
    """
    act_pairs = [(1, 4), (1, 6)]   # 行走vs静坐, 行走vs躺卧
    ch_idx, ch_name = 0, "acc_x"
    slice_times = [0.3, 1.28, 2.2]  # 在 2.56s 窗口内的 3 个时间点 (秒)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for col, (act_a, act_b) in enumerate(act_pairs):
        ax = axes[col]
        colors_a = ["#1565C0", "#2196F3", "#64B5F6"]  # 行走, 深→浅蓝
        colors_b = ["#B71C1C", "#F44336", "#EF9A9A"]  # 静态, 深→浅红

        for act_id, colors in [(act_a, colors_a), (act_b, colors_b)]:
            idxs = np.where(y == act_id)[0]
            win = X[idxs[len(idxs) // 2], :, ch_idx]

            for ti, (t_sec, color) in enumerate(zip(slice_times, colors)):
                start_sample = int(t_sec * FS)
                sub_win = win[start_sample: start_sample + 32]
                if len(sub_win) < 16:
                    continue

                n = len(sub_win)
                mag = np.abs(fft(sub_win)) / n
                mag = mag[: n // 2 + 1]
                mag[1:-1] *= 2
                freqs = fftfreq(n, 1 / FS)[: n // 2 + 1]
                mask = freqs <= 25

                label = f"{活动名称[act_id]} t={t_sec:.1f}s" if ti == 1 else None
                ax.plot(freqs[mask], mag[mask], color=color, linewidth=1.3,
                        alpha=0.85, label=label)

        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("幅度")
        ax.set_title(f"{活动名称[act_a]} vs {活动名称[act_b]} — {ch_name}", fontsize=11)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.set_xlim(0, 25)

    fig.suptitle("频谱时间切片对比 — 3 个时间点的 FFT 频谱叠加",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "13_频谱时间切片对比.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 13_频谱时间切片对比.png")


def main():
    print("加载 UCI HAR 训练集…")
    X, y = load_data()
    print(f"  数据: {X.shape}")

    figure_band_energy_evolution(X, y)
    figure_spectral_slices(X, y)

    print(f"\n完成, 图片保存至 {os.path.abspath(SAVE_DIR)}/")


if __name__ == "__main__":
    main()
