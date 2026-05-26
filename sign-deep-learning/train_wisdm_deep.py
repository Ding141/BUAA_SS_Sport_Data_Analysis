from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.deep_models import build_deep_model
from src.wisdm_data import (
    load_wisdm_cache,
    load_wisdm_fused_windows,
    save_wisdm_cache,
)


def subject_masks(subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = subjects <= 1639
    val_mask = (subjects >= 1640) & (subjects <= 1644)
    test_mask = subjects >= 1645
    return train_mask, val_mask, test_mask


def normalize_by_train(X: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X[train_mask].mean(axis=(0, 2), keepdims=True)
    std = X[train_mask].std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((X - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def make_loader(X: np.ndarray, y: np.ndarray, mask: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X[mask]), torch.from_numpy(y[mask]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * len(yb)
        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())

    if not y_true:
        return 0.0, 0.0
    avg_loss = total_loss / len(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    return avg_loss, accuracy


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deep WISDM accel+gyro activity classifier.")
    parser.add_argument("--data-dir", default="data/wisdm/wisdm-dataset", help="WISDM dataset directory")
    parser.add_argument("--device-type", choices=["phone", "watch"], default="phone", help="WISDM device source")
    parser.add_argument("--cache", default="models/wisdm_deep/fused_windows_phone.npz", help="Window cache path")
    parser.add_argument("--model-out", default="models/wisdm_deep/wisdm_deep_model.pt", help="Model checkpoint path")
    parser.add_argument("--report-dir", default="reports/wisdm_deep", help="Directory to save reports")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--architecture",
        choices=["cnn1d", "cnn_gru", "cnn_bigru", "cnn_lstm", "cnn_bilstm", "transformer", "dual_branch_bigru"],
        default="dual_branch_bigru",
        help="Deep network architecture",
    )
    parser.add_argument("--max-windows-per-class", type=int, default=None, help="Optional quick-run cap")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if cache_path.exists():
        bundle = load_wisdm_cache(cache_path)
    else:
        bundle = load_wisdm_fused_windows(
            Path(args.data_dir),
            device=args.device_type,
            max_windows_per_class=args.max_windows_per_class,
        )
        save_wisdm_cache(bundle, cache_path)

    train_mask, val_mask, test_mask = subject_masks(bundle.subjects)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise RuntimeError(
            "Subject split is empty. Rebuild the cache without an overly small sample cap, "
            "or delete the old cache file and run again."
        )
    X, mean, std = normalize_by_train(bundle.X, train_mask)
    y = bundle.y

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_deep_model(args.architecture, n_channels=X.shape[1], n_classes=len(bundle.label_names)).to(device)

    train_loader = make_loader(X, y, train_mask, args.batch_size, shuffle=True)
    val_loader = make_loader(X, y, val_mask, args.batch_size, shuffle=False)
    test_loader = make_loader(X, y, test_mask, args.batch_size, shuffle=False)

    counts = np.bincount(y[train_mask], minlength=len(bundle.label_names)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = -1.0
    best_state = None
    print(f"Created/loaded {len(y)} windows from WISDM {bundle.device}.")
    print(f"Split sizes: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    print(f"Architecture: {args.architecture}")
    print(f"Training on {device}...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = predict_all(model, test_loader, device)
    test_acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(bundle.label_names))),
        target_names=bundle.label_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(bundle.label_names))))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": args.architecture,
            "label_codes": bundle.label_codes,
            "label_names": bundle.label_names,
            "channels": bundle.channels,
            "window_size": bundle.window_size,
            "sample_rate_hz": bundle.sample_rate_hz,
            "mean": mean,
            "std": std,
            "device_source": bundle.device,
            "test_accuracy": float(test_acc),
        },
        args.model_out,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wisdm_deep_classification_report.txt").write_text(
        f"Model: {args.architecture}\nDevice source: {bundle.device}\n"
        f"Fusion: accelerometer + gyroscope\nTest accuracy: {test_acc:.4f}\n\n{report}",
        encoding="utf-8",
    )
    np.savetxt(report_dir / "wisdm_deep_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    (report_dir / "wisdm_label_map.json").write_text(
        json.dumps(dict(zip(bundle.label_codes, bundle.label_names, strict=True)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved model: {args.model_out}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(report)


if __name__ == "__main__":
    main()
