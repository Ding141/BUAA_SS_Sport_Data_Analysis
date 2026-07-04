r"""SportBUAA 跨数据集活动识别推理流水线。

流程（对应报告第5.2节）:
    1. 数据对齐预处理: 线性插值→50Hz, m/s²→g, 坐标轴翻转对齐
    2. 滑动窗口切分: 128点/窗, 步长64 (50%重叠) → (N, 6, 128)
    3. StandardScaler标准化（使用训练集拟合参数）
    4. FeatureFusionNet推理 → Softmax输出6类概率

用法:
    python predict.py                                # 预测全部样本
    python predict.py --sample walking0              # 预测单个样本
    python predict.py --sample walking0 --detail     # 终端打印逐窗口概率详情

依赖:
    需要 feature_fusion_har.pt（权重）和 feature_fusion_har_meta.json（元数据）
    两个文件需由 deep_learning/train_wisdm_feature_fusion.py 训练生成后放入本目录。
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from features import extract_features
from model import FeatureFusionNet

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "feature_fusion_har.pt")
DEFAULT_META = os.path.join(SCRIPT_DIR, "feature_fusion_har_meta.json")
# SportBUAA 数据目录
SPORTBUAA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "SportBUAA")

LABEL_NAMES = {0: "WALKING", 1: "WALKING_UPSTAIRS", 2: "WALKING_DOWNSTAIRS",
               3: "SITTING", 4: "STANDING", 5: "LAYING"}
LABEL_CN = ["走路", "上楼", "下楼", "静坐", "站立", "躺卧"]

GRAVITY = 9.8  # m/s² → g 换算


# ---------------------------------------------------------------------------
# 步骤1: 数据对齐预处理
# ---------------------------------------------------------------------------
def preprocess_raw(acc_df: pd.DataFrame, gyro_df: pd.DataFrame,
                   target_fs: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """将原始CSV数据预处理对齐。

    CSV列格式: seconds_elapsed, z, y, x

    操作:
        - 按公共时间网格线性插值至 target_fs Hz
        - 坐标轴: (z, y, x) → (x, y, z)，与UCI HAR保持一致
        - 注: SportBUAA 使用 Android TYPE_LINEAR_ACCELERATION，
          已由系统完成重力补偿，数据单位为 g，无需额外单位换算和DC移除
    """
    # 提取时间戳与数据（列顺序: elapsed, z, y, x → x, y, z）
    t_acc = acc_df.iloc[:, 0].values
    data_acc = acc_df.iloc[:, [3, 2, 1]].values.astype(np.float64)  # x, y, z
    t_gyro = gyro_df.iloc[:, 0].values
    data_gyro = gyro_df.iloc[:, [3, 2, 1]].values.astype(np.float64)

    # 公共时间网格（np.arange 可能有浮点精度问题，使用 linspace 确保点数正确）
    t_start = max(t_acc[0], t_gyro[0])
    t_end = min(t_acc[-1], t_gyro[-1])
    duration = t_end - t_start
    n_points = int(np.floor(duration * target_fs)) + 1
    if n_points < 128:
        raise ValueError(f"数据时长不足: {n_points}点 < 128点")
    t_grid = np.linspace(t_start, t_end, n_points)

    # 线性插值
    acc_interp = np.array([np.interp(t_grid, t_acc, data_acc[:, i]) for i in range(3)]).T
    gyro_interp = np.array([np.interp(t_grid, t_gyro, data_gyro[:, i]) for i in range(3)]).T

    return acc_interp.astype(np.float32), gyro_interp.astype(np.float32)


# ---------------------------------------------------------------------------
# 坐标轴对齐
# ---------------------------------------------------------------------------
def detect_and_flip_axes(acc: np.ndarray, gyro: np.ndarray,
                         uci_acc_mean: np.ndarray, uci_gyro_mean: np.ndarray,
                         auto_threshold: float = 0.3) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """检测并翻转坐标轴方向，使之与 UCI HAR 训练集对齐。

    策略：对加速度计和陀螺仪的每个轴，比较 SportBUAA 静态段均值与
    UCI HAR 训练集对应通道均值的符号。符号不一致的轴自动翻转。

    Args:
        acc:  (T, 3) 加速度计插值后数据 (x, y, z)，单位 g
        gyro: (T, 3) 陀螺仪插值后数据 (x, y, z)
        uci_acc_mean:  (3,) UCI HAR 训练集加速度计各通道全局均值
        uci_gyro_mean: (3,) UCI HAR 训练集陀螺仪各通道全局均值
        auto_threshold: 均值绝对值低于此阈值时跳过翻转检测（近零均值无方向性）

    Returns:
        acc_aligned:  (T, 3) 对齐后的加速度计数据
        gyro_aligned: (T, 3) 对齐后的陀螺仪数据
        flips:        记录翻转操作的字符串列表
    """
    acc_aligned = acc.copy()
    gyro_aligned = gyro.copy()
    flips = []
    axis_names = ["x", "y", "z"]

    for i, name in enumerate(axis_names):
        # 加速度计
        sport_mean = float(np.mean(acc[:, i]))
        uci_mean = float(uci_acc_mean[i])
        if abs(sport_mean) > auto_threshold and abs(uci_mean) > auto_threshold:
            if np.sign(sport_mean) != np.sign(uci_mean):
                acc_aligned[:, i] *= -1.0
                flips.append(f"acc_{name}")
        # 陀螺仪同理
        sport_mean_g = float(np.mean(gyro[:, i]))
        uci_mean_g = float(uci_gyro_mean[i])
        if abs(sport_mean_g) > auto_threshold and abs(uci_mean_g) > auto_threshold:
            if np.sign(sport_mean_g) != np.sign(uci_mean_g):
                gyro_aligned[:, i] *= -1.0
                flips.append(f"gyro_{name}")

    if flips:
        print(f"  [坐标轴对齐] 检测到方向不一致，已翻转: {', '.join(flips)}")
    return acc_aligned, gyro_aligned, flips


# ---------------------------------------------------------------------------
# 步骤2: 滑动窗口切分
# ---------------------------------------------------------------------------
def sliding_window(acc: np.ndarray, gyro: np.ndarray,
                   win_len: int = 128, stride: int = 64) -> np.ndarray:
    """将连续信号切分为滑动窗口。

    Returns:
        windows: (N, 6, 128)，通道顺序 acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    """
    n_total = len(acc)
    n_windows = (n_total - win_len) // stride + 1
    if n_windows < 1:
        raise ValueError(f"信号长度{len(acc)}不足以生成一个窗口")
    windows = np.zeros((n_windows, 6, win_len), dtype=np.float32)
    for i in range(n_windows):
        start = i * stride
        windows[i, 0:3, :] = acc[start:start + win_len].T     # acc xyz
        windows[i, 3:6, :] = gyro[start:start + win_len].T    # gyro xyz
    return windows


# ---------------------------------------------------------------------------
# 步骤3+4: 推理
# ---------------------------------------------------------------------------
def predict_sample(windows: np.ndarray, model: FeatureFusionNet,
                   norm: dict, device: torch.device) -> list[dict]:
    """对一组窗口进行推理，返回各窗口的预测结果。

    Returns:
        [{window_idx, pred_label, conf, probs}] 列表
    """
    results = []
    mean_a = np.array(norm.get("accel_mean", [0] * 78), dtype=np.float32)
    std_a  = np.array(norm.get("accel_std",  [1] * 78), dtype=np.float32) + 1e-8
    mean_g = np.array(norm.get("gyro_mean",  [0] * 78), dtype=np.float32)
    std_g  = np.array(norm.get("gyro_std",   [1] * 78), dtype=np.float32) + 1e-8

    for idx in range(len(windows)):
        accel_feats, gyro_feats = extract_features(windows[idx])
        # 步骤3: 标准化
        accel_feats = (accel_feats - mean_a) / std_a
        gyro_feats  = (gyro_feats  - mean_g) / std_g

        xa = torch.tensor(accel_feats, dtype=torch.float32).unsqueeze(0).to(device)
        xg = torch.tensor(gyro_feats, dtype=torch.float32).unsqueeze(0).to(device)

        # 步骤4: 推理
        with torch.no_grad():
            logits = model(xa, xg)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(logits.argmax(dim=1).item())
            conf = float(probs[pred])

        results.append({
            "window_idx": idx,
            "pred_label": LABEL_NAMES[pred],
            "pred_label_cn": LABEL_CN[pred],
            "confidence": conf,
            "probs": {LABEL_NAMES[i]: float(probs[i]) for i in range(6)},
        })
    return results


# ---------------------------------------------------------------------------
# 逐窗口详情打印
# ---------------------------------------------------------------------------
def print_details(results: list[dict], sample_name: str):
    """在终端逐窗口输出各帧预测概率分布。"""
    print(f"\n  {'─' * 58}")
    print(f"  样本 [{sample_name}] 逐窗口推理详情 ({len(results)} 窗口)")
    print(f"  {'─' * 58}")
    for r in results:
        t_sec = r["window_idx"] * 64 / 50  # 步长64 / 50Hz
        pred_cn = r["pred_label_cn"]
        conf = r["confidence"]
        probs_str = "  ".join(
            f"{LABEL_CN[i]}:{r['probs'][LABEL_NAMES[i]]:.3f}" for i in range(6)
        )
        print(f"  t={t_sec:5.1f}s  预测={pred_cn}  置信度={conf:.2%}")
        print(f"           {probs_str}")
    print()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        print("请先运行 deep_learning/train_wisdm_feature_fusion.py 训练模型，")
        print("然后将 feature_fusion_har.pt 和 feature_fusion_har_meta.json 复制到本目录。")
        sys.exit(1)

    # 加载元数据
    with open(args.meta) as f:
        meta = json.load(f)

    # 构建模型
    accel_dim = meta.get("accel_dim", 78)
    gyro_dim = meta.get("gyro_dim", 78)
    model = FeatureFusionNet(accel_dim, gyro_dim, n_classes=6)
    state = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    norm = meta.get("normalization", {})

    # 加载 UCI HAR 训练集通道均值用于坐标轴对齐检测
    uci_acc_mean = np.array(norm.get("uci_acc_channel_mean", [0, 0, 0]), dtype=np.float32)
    uci_gyro_mean = np.array(norm.get("uci_gyro_channel_mean", [0, 0, 0]), dtype=np.float32)

    # 查找样本
    sportbuaa_dir = args.data_dir or SPORTBUAA_DIR
    if not os.path.exists(sportbuaa_dir):
        print(f"错误: SportBUAA 数据目录不存在: {sportbuaa_dir}")
        print("请通过 --data_dir 指定路径")
        sys.exit(1)

    all_dirs = sorted(d for d in os.listdir(sportbuaa_dir)
                      if os.path.isdir(os.path.join(sportbuaa_dir, d)))
    samples = [args.sample] if args.sample else all_dirs
    if not samples:
        print("未找到样本。")
        return

    print(f"{'样本':<28s} {'窗口数':>6s}  {'主导预测':<16s} {'置信度':>8s}")
    print("-" * 68)

    for name in samples:
        group_dir = os.path.join(sportbuaa_dir, name)
        if not os.path.isdir(group_dir):
            print(f"{name:<28s} {'[目录不存在]':>6s}")
            continue

        acc_path = os.path.join(group_dir, "Accelerometer.csv")
        gyro_path = os.path.join(group_dir, "Gyroscope.csv")

        if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
            # 可能数据是预切片的 (sub_dir/ 下直接是128点CSV)
            # 尝试将整个 group_dir 作为一个128点窗口处理
            try:
                acc_df = pd.read_csv(acc_path) if os.path.exists(acc_path) else None
                gyro_df = pd.read_csv(gyro_path) if os.path.exists(gyro_path) else None
            except Exception:
                print(f"{name:<28s} {'[加载失败]':>6s}")
                continue

        try:
            acc_df = pd.read_csv(acc_path)
            gyro_df = pd.read_csv(gyro_path)
        except Exception as e:
            print(f"{name:<28s} {'[加载失败]':>6s}  {e}")
            continue

        # 步骤1+2: 预处理 + 坐标轴对齐 + 滑窗
        try:
            acc_interp, gyro_interp = preprocess_raw(acc_df, gyro_df)
            # 坐标轴对齐：检测并翻转与UCI HAR方向不一致的轴
            acc_aligned, gyro_aligned, flips = detect_and_flip_axes(
                acc_interp, gyro_interp, uci_acc_mean, uci_gyro_mean)
            windows = sliding_window(acc_aligned, gyro_aligned)
        except ValueError as e:
            # 数据可能已经是128点的预切片，直接用一个窗口
            acc_arr = acc_df.iloc[:, [3, 2, 1]].values.T.astype(np.float32)  # x,y,z
            gyro_arr = gyro_df.iloc[:, [3, 2, 1]].values.T.astype(np.float32)
            single_win = np.concatenate([acc_arr, gyro_arr], axis=0)  # (6, N)
            if single_win.shape[1] >= 128:
                single_win = single_win[:, :128]
            else:
                print(f"{name:<28s} {'[数据不足128点]':>6s}")
                continue
            windows = single_win[np.newaxis, ...]  # (1, 6, 128)

        # 步骤3+4: 推理
        results = predict_sample(windows, model, norm, device)

        # 汇总
        if results:
            # 主导预测：出现最多的类别
            from collections import Counter
            preds = [r["pred_label_cn"] for r in results]
            dominant = Counter(preds).most_common(1)[0]
            avg_conf = np.mean([r["confidence"] for r in results])
            print(f"{name:<28s} {len(results):>6d}  {dominant[0]:<16s} {avg_conf:7.2%}")

            # 可选: 打印逐窗口详情
            if args.detail:
                print_details(results, name)
        else:
            print(f"{name:<28s} {'[无结果]':>6s}")

    print("-" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SportBUAA 活动识别推理流水线")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型权重 .pt 文件")
    parser.add_argument("--meta", default=DEFAULT_META, help="元数据 .json 文件")
    parser.add_argument("--sample", default=None, help="单个样本名")
    parser.add_argument("--data_dir", default=None, help="SportBUAA 数据目录")
    parser.add_argument("--detail", action="store_true", help="终端逐窗口打印预测概率详情")
    args = parser.parse_args()
    main(args)
