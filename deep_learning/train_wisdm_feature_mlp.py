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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FeatureMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def subject_masks(subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return subjects <= 1639, ((subjects >= 1640) & (subjects <= 1644)), subjects >= 1645


def normalize(X: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X[mask].mean(axis=0, keepdims=True)
    std = X[mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((X - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def loader(X: np.ndarray, y: np.ndarray, mask: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(X[mask]), torch.from_numpy(y[mask])), batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, data_loader, criterion, optimizer=None) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    truth: list[int] = []
    pred: list[int] = []
    for xb, yb in data_loader:
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(yb)
        truth.extend(yb.numpy().tolist())
        pred.extend(torch.argmax(logits, dim=1).detach().numpy().tolist())
    return total / len(truth), accuracy_score(truth, pred)


def predict(model, data_loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truth: list[int] = []
    pred: list[int] = []
    with torch.no_grad():
        for xb, yb in data_loader:
            logits = model(xb)
            truth.extend(yb.numpy().tolist())
            pred.extend(torch.argmax(logits, dim=1).numpy().tolist())
    return np.asarray(truth), np.asarray(pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an optimized WISDM accel+gyro ARFF-feature deep MLP.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "wisdm-dataset"))
    parser.add_argument("--cache", default="models/wisdm_deep/fused_arff_features_phone.npz")
    parser.add_argument("--model-out", default="models/wisdm_deep/wisdm_feature_mlp.pt")
    parser.add_argument("--report-dir", default="reports/wisdm_deep")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    y = bundle.y

    torch.manual_seed(42)
    np.random.seed(42)
    model = FeatureMLP(X.shape[1], len(bundle.label_names))
    counts = np.bincount(y[train_val_mask], minlength=len(bundle.label_names)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = loader(X, y, train_val_mask, args.batch_size, True)
    test_loader = loader(X, y, test_mask, args.batch_size, False)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

    y_true, y_pred = predict(model, test_loader)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
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
            "model_type": "feature_mlp",
            "label_codes": bundle.label_codes,
            "label_names": bundle.label_names,
            "feature_names": bundle.feature_names,
            "mean": mean,
            "std": std,
            "test_accuracy": float(acc),
            "test_macro_f1": float(macro_f1),
            "source": bundle.source,
        },
        args.model_out,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wisdm_feature_mlp_test_report.txt").write_text(
        f"Model: feature_mlp\nSource: {bundle.source}\n"
        f"Accuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\n\n{report}",
        encoding="utf-8",
    )
    np.savetxt(report_dir / "wisdm_feature_mlp_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    (report_dir / "wisdm_feature_mlp_training.json").write_text(
        json.dumps({"test_accuracy": acc, "test_macro_f1": macro_f1, "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model: {args.model_out}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(report)


if __name__ == "__main__":
    main()
