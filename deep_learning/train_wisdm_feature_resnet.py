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


class ResidualFeatureBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class FeatureResNet(nn.Module):
    """Residual MLP for WISDM ARFF time/frequency features."""

    def __init__(self, n_features: int, n_classes: int, width: int = 384, blocks: int = 3) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(n_features, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.blocks = nn.Sequential(*(ResidualFeatureBlock(width, dropout=0.20) for _ in range(blocks)))
        self.head = nn.Sequential(
            nn.Linear(width, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.input(x)))


def make_loader(X: np.ndarray, y: np.ndarray, mask: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X[mask]), torch.from_numpy(y[mask]))
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
    for xb, yb in data_loader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
        for xb, yb in data_loader:
            logits = model(xb)
            truth.extend(yb.numpy().tolist())
            pred.extend(torch.argmax(logits, dim=1).numpy().tolist())
    return np.asarray(truth), np.asarray(pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a residual feature network on WISDM ARFF features.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "wisdm-dataset"))
    parser.add_argument("--cache", default="models/wisdm_deep/fused_arff_features_phone.npz")
    parser.add_argument("--model-out", default="models/wisdm_deep/wisdm_feature_resnet.pt")
    parser.add_argument("--report-dir", default="reports/wisdm_deep")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if cache_path.exists():
        bundle = load_wisdm_arff_cache(cache_path)
    else:
        bundle = load_wisdm_arff_fused(Path(args.data_dir), device="phone")
        save_wisdm_arff_cache(bundle, cache_path)
    train_mask, val_mask, test_mask = subject_masks(bundle.subjects)
    train_val_mask = train_mask | val_mask
    X, mean, std = normalize(bundle.X, train_mask)
    y = bundle.y

    torch.manual_seed(42)
    np.random.seed(42)
    model = FeatureResNet(X.shape[1], len(bundle.label_names), width=args.width, blocks=args.blocks)

    counts = np.bincount(y[train_mask], minlength=len(bundle.label_names)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights), label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = make_loader(X, y, train_mask, args.batch_size, True)
    val_loader = make_loader(X, y, val_mask, args.batch_size, False)
    test_loader = make_loader(X, y, test_mask, args.batch_size, False)

    best_val_macro_f1 = -1.0
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_true, val_pred = predict(model, val_loader)
        val_acc = accuracy_score(val_true, val_pred)
        val_macro_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
            }
        )
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0 or stale_epochs == args.patience:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                f"val_macro_f1={val_macro_f1:.4f} best={best_val_macro_f1:.4f}"
            )
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_true, test_pred = predict(model, test_loader)
    test_acc = accuracy_score(test_true, test_pred)
    test_macro_f1 = f1_score(test_true, test_pred, average="macro", zero_division=0)

    report = classification_report(
        test_true,
        test_pred,
        labels=list(range(len(bundle.label_names))),
        target_names=bundle.label_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(test_true, test_pred, labels=list(range(len(bundle.label_names))))

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": "feature_resnet",
            "label_codes": bundle.label_codes,
            "label_names": bundle.label_names,
            "feature_names": bundle.feature_names,
            "mean": mean,
            "std": std,
            "source": bundle.source,
            "width": args.width,
            "blocks": args.blocks,
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_macro_f1),
            "selected_by": "highest validation macro F1 on held-out validation subjects",
        },
        model_out,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wisdm_feature_resnet_test_report.txt").write_text(
        f"Model: feature_resnet\nSource: {bundle.source}\n"
        f"Width: {args.width}\nBlocks: {args.blocks}\n"
        f"Selected validation macro F1: {best_val_macro_f1:.4f}\n"
        f"Accuracy: {test_acc:.4f}\nMacro F1: {test_macro_f1:.4f}\n\n{report}",
        encoding="utf-8",
    )
    np.savetxt(report_dir / "wisdm_feature_resnet_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    (report_dir / "wisdm_feature_resnet_training.json").write_text(
        json.dumps(
            {
                "model": "feature_resnet",
                "width": args.width,
                "blocks": args.blocks,
                "best_val_macro_f1": best_val_macro_f1,
                "test_accuracy": test_acc,
                "test_macro_f1": test_macro_f1,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Also train a final model on train+validation for deployment, using the selected architecture.
    final_X, final_mean, final_std = normalize(bundle.X, train_val_mask)
    final_model = FeatureResNet(final_X.shape[1], len(bundle.label_names), width=args.width, blocks=args.blocks)
    final_counts = np.bincount(y[train_val_mask], minlength=len(bundle.label_names)).astype(np.float32)
    final_weights = final_counts.sum() / np.maximum(final_counts, 1.0)
    final_weights = final_weights / final_weights.mean()
    final_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(final_weights), label_smoothing=args.label_smoothing)
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=2e-4)
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=max(1, len(history)))
    final_loader = make_loader(final_X, y, train_val_mask, args.batch_size, True)
    final_test_loader = make_loader(final_X, y, test_mask, args.batch_size, False)
    for _ in range(max(1, len(history))):
        run_epoch(final_model, final_loader, final_criterion, final_optimizer)
        final_scheduler.step()
    final_true, final_pred = predict(final_model, final_test_loader)
    final_acc = accuracy_score(final_true, final_pred)
    final_macro_f1 = f1_score(final_true, final_pred, average="macro", zero_division=0)
    torch.save(
        {
            "model_state": final_model.state_dict(),
            "model_type": "feature_resnet",
            "label_codes": bundle.label_codes,
            "label_names": bundle.label_names,
            "feature_names": bundle.feature_names,
            "mean": final_mean,
            "std": final_std,
            "source": bundle.source,
            "width": args.width,
            "blocks": args.blocks,
            "test_accuracy": float(final_acc),
            "test_macro_f1": float(final_macro_f1),
            "final_training": "train+validation subjects after validation-based architecture selection",
        },
        model_out.with_name("wisdm_feature_resnet_final.pt"),
    )
    print(f"Saved validation-selected model: {model_out}")
    print(f"Validation-selected test accuracy: {test_acc:.4f}")
    print(f"Validation-selected test macro F1: {test_macro_f1:.4f}")
    print(f"Saved train+validation final model: {model_out.with_name('wisdm_feature_resnet_final.pt')}")
    print(f"Final train+validation test accuracy: {final_acc:.4f}")
    print(f"Final train+validation test macro F1: {final_macro_f1:.4f}")
    print(report)


if __name__ == "__main__":
    main()
