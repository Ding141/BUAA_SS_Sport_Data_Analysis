from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.wisdm_arff import load_wisdm_arff_cache, load_wisdm_arff_fused, save_wisdm_arff_cache
from train_wisdm_feature_mlp import normalize, subject_masks

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SensorFeatureBranch(nn.Module):
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
    """Two-branch feature network for accel/gyro ARFF features."""

    def __init__(self, accel_dim: int, gyro_dim: int, n_classes: int) -> None:
        super().__init__()
        self.accel_branch = SensorFeatureBranch(accel_dim)
        self.gyro_branch = SensorFeatureBranch(gyro_dim)
        fused_dim = 80 * 4
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


def split_accel_gyro(X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accel_idx = np.asarray([i for i, name in enumerate(feature_names) if name.startswith("accel_")], dtype=np.int64)
    gyro_idx = np.asarray([i for i, name in enumerate(feature_names) if name.startswith("gyro_")], dtype=np.int64)
    if len(accel_idx) == 0 or len(gyro_idx) == 0:
        raise ValueError("Expected feature names to contain accel_ and gyro_ prefixes")
    return X[:, accel_idx], X[:, gyro_idx], accel_idx, gyro_idx


def make_loader(
    Xa: np.ndarray,
    Xg: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(Xa[mask]), torch.from_numpy(Xg[mask]), torch.from_numpy(y[mask]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    truth: list[int] = []
    pred: list[int] = []
    for xa, xg, yb in data_loader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xa, xg)
        loss = criterion(logits, yb)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(yb)
        truth.extend(yb.numpy().tolist())
        pred.extend(torch.argmax(logits, dim=1).detach().numpy().tolist())
    return total_loss / len(truth), accuracy_score(truth, pred)


def predict(model: nn.Module, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truth: list[int] = []
    pred: list[int] = []
    with torch.no_grad():
        for xa, xg, yb in data_loader:
            logits = model(xa, xg)
            truth.extend(yb.numpy().tolist())
            pred.extend(torch.argmax(logits, dim=1).numpy().tolist())
    return np.asarray(truth), np.asarray(pred)


def evaluate_and_save(
    model: nn.Module,
    loader: DataLoader,
    label_names: list[str],
    report_path: Path,
    cm_path: Path,
    header: str,
) -> tuple[float, float]:
    y_true, y_pred = predict(model, loader)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    report_path.write_text(f"{header}\nAccuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\n\n{report}", encoding="utf-8")
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")
    return acc, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a two-branch accel/gyro feature fusion network.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "wisdm-dataset"))
    parser.add_argument("--cache", default="models/wisdm_deep/fused_arff_features_phone.npz")
    parser.add_argument("--model-out", default="models/wisdm_deep/wisdm_feature_fusion.pt")
    parser.add_argument("--report-dir", default="reports/wisdm_deep")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if cache_path.exists():
        bundle = load_wisdm_arff_cache(cache_path)
    else:
        bundle = load_wisdm_arff_fused(Path(args.data_dir), device="phone")
        save_wisdm_arff_cache(bundle, cache_path)
    train_mask, val_mask, test_mask = subject_masks(bundle.subjects)
    train_val_mask = train_mask | val_mask
    X, mean, std = normalize(bundle.X, train_val_mask)
    Xa, Xg, accel_idx, gyro_idx = split_accel_gyro(X, bundle.feature_names)
    y = bundle.y

    torch.manual_seed(42)
    np.random.seed(42)
    model = FeatureFusionNet(Xa.shape[1], Xg.shape[1], len(bundle.label_names))
    counts = np.bincount(y[train_val_mask], minlength=len(bundle.label_names)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights), label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = make_loader(Xa, Xg, y, train_val_mask, args.batch_size, True)
    test_loader = make_loader(Xa, Xg, y, test_mask, args.batch_size, False)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    acc, macro_f1 = evaluate_and_save(
        model,
        test_loader,
        bundle.label_names,
        report_dir / "wisdm_feature_fusion_test_report.txt",
        report_dir / "wisdm_feature_fusion_confusion_matrix.csv",
        "Model: feature_fusion\nSource: phone accel+gyro ARFF\nTraining data: train+validation subjects",
    )

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": "feature_fusion",
            "label_codes": bundle.label_codes,
            "label_names": bundle.label_names,
            "feature_names": bundle.feature_names,
            "accel_indices": accel_idx,
            "gyro_indices": gyro_idx,
            "mean": mean,
            "std": std,
            "source": bundle.source,
            "test_accuracy": float(acc),
            "test_macro_f1": float(macro_f1),
        },
        model_out,
    )
    (report_dir / "wisdm_feature_fusion_training.json").write_text(
        json.dumps(
            {"model": "feature_fusion", "test_accuracy": acc, "test_macro_f1": macro_f1, "history": history},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model: {model_out}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
