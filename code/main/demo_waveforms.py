"""
UCI HAR Demo: 6 种动作的时域曲线
===============================
从训练集里每种动作各取一条样例窗口，画加速度计三轴时域波形。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FS = 50          # Hz
N_SAMPLES = 128  # 2.56 s
DATA_DIR = "UCI HAR Dataset"
SAVE_DIR = os.path.join("figures", "演示")
os.makedirs(SAVE_DIR, exist_ok=True)

ACTIVITY = {
    1: "Walking", 2: "Walking Upstairs", 3: "Walking Downstairs",
    4: "Sitting", 5: "Standing", 6: "Laying",
}
COLORS = {"body_acc_x": "#F44336", "body_acc_y": "#4CAF50", "body_acc_z": "#2196F3"}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8,
    "font.sans-serif": ["Arial Unicode MS", "SimHei", "Heiti SC", "STHeiti"],
    "axes.unicode_minus": False,
})


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


def main():
    print("加载 UCI HAR 数据…")
    X, y = load_data()
    t = np.arange(N_SAMPLES) / FS

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for act_id in range(1, 7):
        row, col = (act_id - 1) // 3, (act_id - 1) % 3
        ax = axes[row][col]

        # 取该活动中间位置的窗口作为代表
        idxs = np.where(y == act_id)[0]
        sample_idx = idxs[len(idxs) // 2]

        for ch_idx, (ch_name, ch_color) in enumerate(zip(
            ["body_acc_x", "body_acc_y", "body_acc_z"],
            ["#F44336", "#4CAF50", "#2196F3"],
        )):
            ax.plot(t, X[sample_idx, :, ch_idx],
                    color=ch_color, linewidth=0.8, label=ch_name)

        ax.set_title(f"{act_id}. {ACTIVITY[act_id]}", fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (g)")
        ax.set_xlim(0, 2.56)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("UCI HAR — Time Domain Waveforms (body_acc x/y/z)\nOne Example Window per Activity (Train Set)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "demo_6act_waveforms.png"))
    plt.close(fig)
    print(f"已保存: {SAVE_DIR}/demo_6act_waveforms.png")

    # 终端摘要
    print("\n各动作时域特征对比 (body_acc_x):")
    print(f"  {'动作':<22s} {'均值':>8s} {'方差':>8s} {'峰峰值':>8s} {'Max':>8s} {'Min':>8s}")
    print("  " + "─" * 60)
    for act_id in range(1, 7):
        idxs = np.where(y == act_id)[0]
        idx = idxs[len(idxs) // 2]
        sig = X[idx, :, 0]
        print(f"  {ACTIVITY[act_id]:<22s} {np.mean(sig):>8.3f} {np.var(sig):>8.3f} "
              f"{np.ptp(sig):>8.3f} {np.max(sig):>8.3f} {np.min(sig):>8.3f}")


if __name__ == "__main__":
    main()
