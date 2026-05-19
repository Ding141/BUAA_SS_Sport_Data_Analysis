"""
特征重要性报告
==============
对全部 258 维特征, 计算并排名:
  - MI (互信息): 单特征与标签的依赖强度, 捕捉线性+非线性关系
  - RF Importance (Gini): 随机森林中该特征用于分裂的加权频次
生成排序表格, 附带每维的物理含义说明。
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import entropy as stats_entropy, skew, kurtosis, iqr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import os
import warnings
warnings.filterwarnings("ignore")

FS = 50
N_SAMPLES = 128
SUB_WIN = 32
SUB_STRIDE = 16
N_SUB = (N_SAMPLES - SUB_WIN) // SUB_STRIDE + 1
DATA_DIR = "UCI HAR Dataset"

# ── 特征说明字典 ──
频域说明 = {
    "PeakFreq":     "幅度谱峰值对应的频率 (Hz), 反映主导运动频率",
    "MeanFreq":     "频谱质心, 幅度加权平均频率 (Hz), 能量分布的平衡点",
    "MedianFreq":   "累计能量达50%时的截止频率 (Hz), 比质心更稳健",
    "Energy":       "幅度谱平方和, 反映该通道的总运动强度",
    "Entropy":      "归一化幅度谱的信息熵, 越高频谱越'杂乱', 越低越'纯净'",
    "BandLow":      "0~5Hz 宽频带能量 (Nyquist=25Hz的0-20%), 静态动作能量集中于此",
    "BandMid":      "5~15Hz 宽频带能量 (Nyquist的20-60%), 中等速度动作",
    "BandHigh":     "15~25Hz 宽频带能量 (Nyquist的60-100%), 高频冲击/噪声",
    "PeakMag":      "幅度谱的最大值, 主频的强度",
    "Spread":       "频谱扩散度 (围绕质心的二阶矩), 频谱有多'宽'",
    "Skewness_f":   "频谱偏度 (三阶矩), >0表示能量偏高频, <0偏低频",
    "Kurtosis_f":   "频谱峰度 (四阶矩), 越高频谱主峰越尖锐突出",
    "Flatness":     "谱平坦度 = 几何平均/算术平均, 0=纯正弦波, 1=白噪声",
    "Crest":        "谱峰因子 = max/mean, 主频相对平均水平的突出程度",
    "Rolloff75":    "累计能量75%的截止频率 (Hz), 能量集中度的度量",
    "Rolloff90":    "累计能量90%的截止频率 (Hz), 同Rolloff75但更保守",
    "Rolloff95":    "累计能量95%的截止频率 (Hz), 几乎捕获全部信号能量",
    "B_0-1Hz":      "0~1Hz 细频带能量, DC附近——静坐/站立/躺卧的全部能量几乎在此",
    "B_1-3Hz":      "1~3Hz 细频带能量, 步态基频区——行走约1.5-2Hz主峰所在",
    "B_3-5Hz":      "3~5Hz 细频带能量, 步态谐波区——上下楼时此频带能量升高",
    "B_5-8Hz":      "5~8Hz 细频带能量, 中频过渡带——快走/慢跑谐波",
    "B_8-12Hz":     "8~12Hz 细频带能量, 中高频——剧烈动作或传感器噪声",
    "B_12-18Hz":    "12~18Hz 细频带能量, 高频段——主要反映冲击锐度",
    "B_18-25Hz":    "18~25Hz 细频带能量, Nyquist上限附近——通常是噪声主导",
}

时域说明 = {
    "Mean":         "信号均值, 反映传感器的静态偏置 (重力分量)",
    "Var":          "信号方差, 反映信号围绕均值的波动强度 = 交流能量",
    "PTP":          "峰峰值 (max-min), 反映窗口内最大冲击幅度",
    "ZCR":          "过零率, 信号穿越均值线的频率——粗略估计主频",
    "RMS":          "均方根值 = sqrt(mean(signal²)), 信号的'有效值'",
    "Skewness_t":   "时域偏度, >0表示尖峰向上, <0表示尖峰向下, 上下楼差异大",
    "Kurtosis_t":   "时域峰度, >0表示有尖锐冲击峰, ≈0接近正态分布, <0表示平坦",
    "IQR":          "四分位距 (Q3−Q1), 比方差更稳健的离散度量, 不受极端值影响",
    "Median":       "信号中位数, 比均值更稳健的中心趋势度量",
    "Max":          "信号最大值, 正向冲击峰值",
    "Min":          "信号最小值, 反向冲击峰值 (绝对值)",
}

滑动窗说明 = {
    "SW_KurtT_mean":    "7个子窗时域峰度的均值 —— 整体冲击'尖度'水平",
    "SW_KurtT_std":     "7个子窗时域峰度的标准差 —— 峰度在步态周期内的波动幅度",
    "SW_KurtT_kurt":    "7个子窗时域峰度的峰度 —— 峰度分布的'峰值度', 高值表示有极端冲击子窗",
    "SW_KurtT_ptp":     "7个子窗时域峰度的极差 —— 最冲击子窗与最平缓子窗的差异",
    "SW_KurtF_mean":    "7个子窗频域峰度的均值 —— 频谱整体'尖锐度'水平",
    "SW_KurtF_std":     "7个子窗频域峰度的标准差 —— 频谱尖锐度在步态周期内的变化",
    "SW_KurtF_kurt":    "7个子窗频域峰度的峰度 —— 频域特征在时序上的异常程度",
    "SW_KurtF_ptp":     "7个子窗频域峰度的极差 —— 频谱最尖锐时刻与最平坦时刻的跨度",
    "SW_PeakF_mean":    "7个子窗主频的均值 —— 窗口内主导频率的平均水平",
    "SW_PeakF_std":     "7个子窗主频的标准差 —— 主导频率在窗口内的稳定性",
    "SW_PeakF_kurt":    "7个子窗主频的峰度 —— 主频时序的异常程度",
    "SW_PeakF_ptp":     "7个子窗主频的极差 —— 频率漂移范围",
    "SW_Centroid_mean": "7个子窗频谱质心的均值 —— 能量中心频率的平均位置",
    "SW_Centroid_std":  "7个子窗频谱质心的标准差 —— 能量中心频率的时序稳定性",
    "SW_Centroid_kurt": "7个子窗频谱质心的峰度 —— 能量迁移的异常程度",
    "SW_Centroid_ptp":  "7个子窗频谱质心的极差 —— 能量在频域的最大迁移范围",
}

通道名 = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]

# ═══════════════════════════════════════════════════════════════
#  数据加载 + 特征提取 (同 V5)
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


def 滑动窗特征(信号):
    kt_list, kf_list, pf_list, cen_list = [], [], [], []
    for i in range(N_SUB):
        start = i * SUB_STRIDE
        sub = 信号[start:start + SUB_WIN]
        kt_list.append(kurtosis(sub))
        freqs, mag = fft_spectrum(sub)
        total = np.sum(mag) + 1e-12
        mask = freqs >= 0.5
        mag_valid = mag[mask]
        kf_list.append(kurtosis(mag_valid) if len(mag_valid) > 4 else 0.0)
        pf_list.append(freqs[np.argmax(mag)])
        cen_list.append(np.sum(freqs * mag) / total)

    feats = []
    for arr in [np.array(kt_list), np.array(kf_list), np.array(pf_list), np.array(cen_list)]:
        feats.append(np.mean(arr))
        feats.append(np.std(arr))
        feats.append(kurtosis(arr) if np.std(arr) > 1e-8 else 0.0)
        feats.append(np.ptp(arr))
    return feats


def build_feature_names():
    """构建 258 维特征名列表 + 说明映射。"""
    names, descs, types_, chs = [], [], [], []

    # 静态频域特征名
    静态频域名 = list(频域说明.keys())
    静态时域名 = list(时域说明.keys())
    滑动窗特征名 = list(滑动窗说明.keys())

    for ch in 通道名:
        for fn in 静态频域名:
            names.append(f"{ch}_{fn}")
            descs.append(频域说明[fn])
            types_.append("频域")
            chs.append(ch)
        for fn in 静态时域名:
            names.append(f"{ch}_{fn}")
            descs.append(时域说明[fn])
            types_.append("时域")
            chs.append(ch)

    # 滑动窗特征 (仅 AccX/Y/Z)
    for ch in ["AccX", "AccY", "AccZ"]:
        for fn in 滑动窗特征名:
            names.append(f"{ch}_{fn}")
            descs.append(滑动窗说明[fn])
            types_.append("滑动窗")
            chs.append(ch)

    return names, descs, types_, chs


def extract_all_features(raw_data, verbose=True):
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
            if ch < 3:
                row.extend(滑动窗特征(signal))
        all_rows.append(row)
        if verbose and (i + 1) % 2000 == 0:
            print(f"  特征提取: {i + 1}/{n_windows}")
    return np.array(all_rows, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("  特征重要性报告: 258 维特征 MI + RF Gini 排名")
    print("═" * 70)

    # 加载
    print("\n📂 加载数据…")
    X_raw, y = load_data()

    # 特征提取
    print("🔧 提取 258 维特征…")
    X_train = extract_all_features(X_raw["train"])
    print(f"  X_train: {X_train.shape}")

    # 构建特征名
    names, descs, types_, chs = build_feature_names()
    print(f"  特征名数: {len(names)}, 实际维度: {X_train.shape[1]}")
    assert len(names) == X_train.shape[1], f"特征名数({len(names)}) ≠ 维度({X_train.shape[1]})"

    # MI
    print("\n📊 计算互信息 (MI)…")
    mi_scores = mutual_info_classif(X_train, y["train"], random_state=42)

    # RF 重要性
    print("🌲 训练随机森林 (200棵树) 计算 Gini 重要性…")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y["train"])
    rf_scores = rf.feature_importances_

    # 按 MI 排序
    order_mi = np.argsort(mi_scores)[::-1]

    # ── 终端表格 ──
    print("\n" + "═" * 110)
    print("  特征重要性排名 (按 MI 降序, 前 60 维)")
    print("═" * 110)
    print(f"  {'排名':<5s} {'特征名':<32s} {'通道':<8s} {'类型':<6s} {'MI':<8s} {'RF Gini':<8s} {'说明'}")
    print("  " + "─" * 106)

    for rank, idx in enumerate(order_mi[:60], 1):
        print(f"  {rank:<5d} {names[idx]:<32s} {chs[idx]:<8s} {types_[idx]:<6s} "
              f"{mi_scores[idx]:>7.4f} {rf_scores[idx]:>7.4f}  {descs[idx]}")

    # ── 统计摘要 ──
    print("\n" + "═" * 70)
    print("  统计摘要")
    print("═" * 70)

    # 零 MI 特征
    zero_mi = np.sum(mi_scores < 1e-6)
    print(f"  MI≈0 的特征数 (完全无区分力): {zero_mi}")
    if zero_mi > 0:
        print(f"  列表: {[names[i] for i in np.where(mi_scores < 1e-6)[0]]}")

    # 各类型占比
    for t in ["频域", "时域", "滑动窗"]:
        mask_t = np.array([tt == t for tt in types_])
        count = np.sum(mask_t)
        if count > 0:
            avg_mi = np.mean(mi_scores[mask_t])
            avg_rf = np.mean(rf_scores[mask_t])
            top30_count = sum(1 for i in order_mi[:30] if types_[i] == t)
            print(f"  {t}特征: {count}维 | MI均值={avg_mi:.4f} | RF均值={avg_rf:.4f} | Top30占{top30_count}维")

    # 各通道贡献
    print(f"\n  各通道 Top-30 占比:")
    for ch in 通道名:
        cnt = sum(1 for i in order_mi[:30] if chs[i] == ch)
        print(f"    {ch}: {cnt}/30")

    # 滑动窗特征排名
    sw_indices = [i for i in order_mi if types_[i] == "滑动窗"]
    print(f"\n  滑动窗特征排名情况 (共{len(sw_indices)}维):")
    for rank, idx in enumerate(sw_indices[:15], 1):
        global_rank = list(order_mi).index(idx) + 1
        print(f"    全局第{global_rank}名: {names[idx]:35s} MI={mi_scores[idx]:.4f} RF={rf_scores[idx]:.4f}")

    # ── 导出 CSV ──
    csv_path = os.path.join("figures", "分类器对比", "特征重要性排名.csv")
    os.makedirs(os.path.join("figures", "分类器对比"), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("排名,特征名,通道,类型,MI得分,RF_Gini得分,说明\n")
        for rank, idx in enumerate(order_mi, 1):
            f.write(f"{rank},{names[idx]},{chs[idx]},{types_[idx]},"
                    f"{mi_scores[idx]:.6f},{rf_scores[idx]:.6f},{descs[idx]}\n")
    print(f"\n  📄 CSV 已导出: {csv_path}")

    print("\n" + "═" * 70)
    print("  完成")
    print("═" * 70)


if __name__ == "__main__":
    main()
