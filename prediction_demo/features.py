"""时频域特征提取：从 (6, 128) 传感器窗口提取每通道26维特征。

每通道提取 13 时域 + 13 频域 → 26维/通道。
加速度计3通道 → 78维，陀螺仪3通道 → 78维。
"""
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import entropy as stats_entropy, skew, kurtosis, iqr


FS = 50  # UCI HAR 采样率


def _fft_spectrum(signal: np.ndarray):
    """单边幅度谱，返回 (freqs, mag)。"""
    n = len(signal)
    vals = fft(signal)
    mag = np.abs(vals) / n
    mag = mag[: n // 2 + 1]
    mag[1:-1] *= 2
    freqs = fftfreq(n, 1 / FS)[: n // 2 + 1]
    return freqs, mag


def _channel_features(signal: np.ndarray) -> np.ndarray:
    """对单通道128点信号提取26维特征。"""
    feats = []
    eps = 1e-12

    # ---- 时域 13 维 ----
    feats.append(np.mean(signal))
    feats.append(np.std(signal))
    feats.append(np.var(signal))
    feats.append(np.max(signal))
    feats.append(np.min(signal))
    feats.append(np.ptp(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    feats.append(rms)
    feats.append(np.sum(np.diff(np.signbit(signal))) / len(signal))  # ZCR
    feats.append(iqr(signal))
    feats.append(skew(signal))
    feats.append(kurtosis(signal))
    feats.append(np.median(signal))
    feats.append(np.mean(np.abs(signal - np.mean(signal))))  # MAD

    # ---- 频域 13 维 ----
    freqs, mag = _fft_spectrum(signal)
    total_mag = np.sum(mag)

    feats.append(freqs[np.argmax(mag)])              # 主频
    if total_mag > eps:
        feats.append(np.sum(freqs * mag) / total_mag)  # 频谱质心
        cum = np.cumsum(mag)
        feats.append(freqs[np.searchsorted(cum, total_mag / 2)])  # 中位数频率
        feats.append(np.sum(mag ** 2))                # 频谱能量
        prob = mag / total_mag
        feats.append(stats_entropy(prob + eps))       # 谱熵
        # 5段频带能量 (DC-2, 2-5, 5-10, 10-15, 15-25 Hz)
        bands = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 25)]
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            feats.append(np.sum(mag[mask] ** 2))
        # 谱形状统计
        feats.append(np.sum(((freqs - feats[-6]) ** 2) * mag) / total_mag)  # 扩散度
        feats.append(skew(mag))                         # 频谱偏度
        feats.append(kurtosis(mag))                     # 频谱峰度
    else:
        feats.extend([0.0] * 13)

    assert len(feats) == 26, f"期望26维，实际{len(feats)}维"
    return np.array(feats, dtype=np.float32)


def extract_features(window: np.ndarray):
    """从 (6, 128) 传感器窗口提取特征。

    Args:
        window: shape (6, 128)，通道顺序: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

    Returns:
        accel_feats: (78,) 加速度计特征
        gyro_feats:  (78,) 陀螺仪特征
    """
    assert window.shape == (6, 128), f"期望 (6,128)，实际 {window.shape}"
    accel_feats = np.concatenate([_channel_features(window[i, :]) for i in range(3)])
    gyro_feats = np.concatenate([_channel_features(window[i, :]) for i in range(3, 6)])
    return accel_feats, gyro_feats
