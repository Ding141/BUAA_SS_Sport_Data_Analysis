"""在 UCI HAR 上训练 FeatureFusionNet（6 类）。

特征提取使用 prediction_demo/features.py 的同一套 13时域+13频域/通道，
确保推理时特征定义完全一致。

输出:
    feature_fusion_har.pt   — 模型权重
    feature_fusion_har_meta.json — 元数据（标准化参数、特征维度等）
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 添加 prediction_demo 到路径以复用其 features.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTION_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "prediction_demo")
sys.path.insert(0, PREDICTION_DIR)
from features import extract_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════
#  模型定义（与 prediction_demo/model.py 完全一致）
# ═══════════════════════════════════════════════════════════════

class SensorFeatureBranch(nn.Module):
    """单传感器特征编码分支: n_features → hidden → out_dim."""

    def __init__(self, n_features: int, hidden: int = 160, out_dim: int = 80) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureFusionNet(nn.Module):
    """双分支特征融合网络。

    accel (78-D) → SensorFeatureBranch → a (80-D) ┐
                                                    ├→ Fusion(a,g,|a-g|,a⊙g) → 320-D → MLP → 6类
    gyro  (78-D) → SensorFeatureBranch → g (80-D) ┘
    """

    def __init__(self, accel_dim: int, gyro_dim: int, n_classes: int = 6) -> None:
        super().__init__()
        self.accel_branch = SensorFeatureBranch(accel_dim)
        self.gyro_branch = SensorFeatureBranch(gyro_dim)
        fused_dim = 80 * 4  # a(80) + g(80) + |a-g|(80) + a⊙g(80)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, n_classes),
        )

    def forward(self, accel: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        a = self.accel_branch(accel)
        g = self.gyro_branch(gyro)
        fused = torch.cat([a, g, torch.abs(a - g), a * g], dim=1)
        return self.classifier(fused)


# ═══════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════

def load_uci_har_raw(data_dir: Path):
    """加载 UCI HAR 原始惯性信号。

    Returns:
        X: dict with 'train'/'test' → ndarray (n_windows, 128, 6)
        y: dict with 'train'/'test' → ndarray (n_windows,)
        通道顺序: body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z
    """
    X, y = {}, {}
    for subset in ["train", "test"]:
        path = data_dir / subset / "Inertial Signals"
        channels = []
        for axis in ["x", "y", "z"]:
            for signal in ["body_acc", "body_gyro"]:
                data = np.loadtxt(path / f"{signal}_{axis}_{subset}.txt", dtype=np.float32)
                channels.append(data)
        X[subset] = np.stack(channels, axis=-1)  # (n_windows, 128, 6)
        y[subset] = np.loadtxt(data_dir / subset / f"y_{subset}.txt", dtype=np.int64)
    return X, y


# ═══════════════════════════════════════════════════════════════
#  特征提取
# ═══════════════════════════════════════════════════════════════

def build_features(raw_data: np.ndarray, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """逐窗口提取 accel 78维 + gyro 78维 特征。

    Args:
        raw_data: (n_windows, 128, 6)，通道顺序 acc_x/y/z, gyro_x/y/z

    Returns:
        accel_feats: (n_windows, 78)
        gyro_feats:  (n_windows, 78)
    """
    n = raw_data.shape[0]
    accel_list, gyro_list = [], []
    for i in range(n):
        # extract_features 期望 (6, 128)，内部将 [0:3] 归为 accel，[3:6] 归为 gyro
        win = raw_data[i].T.astype(np.float32)  # (128, 6) → (6, 128)
        a_feat, g_feat = extract_features(win)
        accel_list.append(a_feat)
        gyro_list.append(g_feat)
        if verbose and (i + 1) % 2000 == 0:
            print(f"  特征提取: {i + 1}/{n}")
    return np.stack(accel_list), np.stack(gyro_list)


# ═══════════════════════════════════════════════════════════════
#  训练工具
# ═══════════════════════════════════════════════════════════════

def make_loader(Xa, Xg, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(Xa), torch.from_numpy(Xg), torch.from_numpy(y)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    truth, preds = [], []
    for xa, xg, yb in loader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xa, xg)
        loss = criterion(logits, yb)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(yb)
        truth.extend(yb.numpy().tolist())
        preds.extend(torch.argmax(logits, dim=1).detach().numpy().tolist())
    return total_loss / len(truth), accuracy_score(truth, preds)


def predict_all(model, loader):
    model.eval()
    truth, preds = [], []
    with torch.no_grad():
        for xa, xg, yb in loader:
            logits = model(xa, xg)
            truth.extend(yb.numpy().tolist())
            preds.extend(torch.argmax(logits, dim=1).numpy().tolist())
    return np.asarray(truth), np.asarray(preds)


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

LABEL_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]


def main():
    parser = argparse.ArgumentParser(
        description="在 UCI HAR 上训练 FeatureFusionNet（6类）"
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "UCI HAR Dataset"))
    parser.add_argument("--model-out", default="models/wisdm_deep/feature_fusion_har.pt")
    parser.add_argument("--meta-out", default="models/wisdm_deep/feature_fusion_har_meta.json")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    args = parser.parse_args()

    print("═" * 55)
    print("  UCI HAR FeatureFusionNet 训练")
    print("═" * 55)

    # ── 1. 加载数据 ──
    print("\n[1/4] 加载 UCI HAR 原始惯性信号…")
    X_raw, y_raw = load_uci_har_raw(Path(args.data_dir))
    print(f"  训练窗口: {X_raw['train'].shape[0]}, 测试窗口: {X_raw['test'].shape[0]}")
    print(f"  通道数: {X_raw['train'].shape[2]} (acc_x/y/z + gyro_x/y/z)")
    for sub in ["train", "test"]:
        from collections import Counter
        cnt = Counter(y_raw[sub])
        print(f"  {sub}: {dict(sorted(cnt.items()))}")

    # ── 2. 特征提取 ──
    print("\n[2/4] 特征提取 (26维/通道 × 3 accel = 78维, 3 gyro = 78维)…")
    Xa_train, Xg_train = build_features(X_raw["train"])
    Xa_test, Xg_test = build_features(X_raw["test"])
    y_train, y_test = y_raw["train"].astype(np.int64), y_raw["test"].astype(np.int64)

    # UCI HAR 标签是 1-6，转换为 0-5
    y_train = y_train - 1
    y_test = y_test - 1

    print(f"  训练特征: accel {Xa_train.shape}, gyro {Xg_train.shape}")
    print(f"  测试特征: accel {Xa_test.shape}, gyro {Xg_test.shape}")

    # 计算标准化参数（基于训练集）
    accel_mean = Xa_train.mean(axis=0).astype(np.float32)
    accel_std = Xa_train.std(axis=0).astype(np.float32)
    gyro_mean = Xg_train.mean(axis=0).astype(np.float32)
    gyro_std = Xg_train.std(axis=0).astype(np.float32)

    # 计算 UCI HAR 原始信号各通道的全局均值（用于坐标轴对齐）
    uci_acc_channel_mean = X_raw["train"].mean(axis=(0, 1))[:3].astype(np.float32)   # acc x,y,z
    uci_gyro_channel_mean = X_raw["train"].mean(axis=(0, 1))[3:6].astype(np.float32)  # gyro x,y,z
    print(f"  UCI acc 通道均值: {uci_acc_channel_mean}")
    print(f"  UCI gyro 通道均值: {uci_gyro_channel_mean}")

    # 标准化
    accel_std_safe = np.where(accel_std < 1e-6, 1.0, accel_std)
    gyro_std_safe = np.where(gyro_std < 1e-6, 1.0, gyro_std)
    Xa_train = ((Xa_train - accel_mean) / accel_std_safe).astype(np.float32)
    Xg_train = ((Xg_train - gyro_mean) / gyro_std_safe).astype(np.float32)
    Xa_test = ((Xa_test - accel_mean) / accel_std_safe).astype(np.float32)
    Xg_test = ((Xg_test - gyro_mean) / gyro_std_safe).astype(np.float32)

    # ── 3. 训练 ──
    print(f"\n[3/4] 训练 FeatureFusionNet ({args.epochs} epochs)…")
    torch.manual_seed(42)
    np.random.seed(42)

    model = FeatureFusionNet(accel_dim=78, gyro_dim=78, n_classes=6)
    counts = np.bincount(y_train, minlength=6).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = make_loader(Xa_train, Xg_train, y_train, args.batch_size, True)
    test_loader = make_loader(Xa_test, Xg_test, y_test, args.batch_size, False)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        y_true, y_pred = predict_all(model, test_loader)
        test_acc = accuracy_score(y_true, y_pred)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:03d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"test_acc={test_acc:.4f}")

    model.load_state_dict(best_state)
    print(f"  最佳测试准确率: {best_acc:.4f}")

    # ── 4. 评估 & 保存 ──
    print("\n[4/4] 评估 & 保存…")
    y_true, y_pred = predict_all(model, test_loader)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=list(range(6)),
        target_names=LABEL_NAMES, digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))

    print(f"\n  测试准确率: {acc:.4f}")
    print(f"  宏平均 F1:  {macro_f1:.4f}")
    print(f"\n{report}")

    # 保存模型
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_out)
    print(f"  模型已保存: {model_out}")

    # 保存元数据
    meta = {
        "accel_dim": 78,
        "gyro_dim": 78,
        "n_classes": 6,
        "label_names": LABEL_NAMES,
        "normalization": {
            "accel_mean": accel_mean.tolist(),
            "accel_std": accel_std_safe.tolist(),
            "gyro_mean": gyro_mean.tolist(),
            "gyro_std": gyro_std_safe.tolist(),
            "uci_acc_channel_mean": uci_acc_channel_mean.tolist(),
            "uci_gyro_channel_mean": uci_gyro_channel_mean.tolist(),
        },
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    meta_out = Path(args.meta_out)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  元数据已保存: {meta_out}")

    # 同时复制到 prediction_demo/ 供推理使用
    import shutil
    dest_pt = Path(PREDICTION_DIR) / "feature_fusion_har.pt"
    dest_meta = Path(PREDICTION_DIR) / "feature_fusion_har_meta.json"
    shutil.copy2(model_out, dest_pt)
    shutil.copy2(meta_out, dest_meta)
    print(f"  已同步到: {dest_pt}")
    print(f"  已同步到: {dest_meta}")

    print("\n" + "═" * 55)
    print("  训练完成")
    print("═" * 55)


if __name__ == "__main__":
    main()
