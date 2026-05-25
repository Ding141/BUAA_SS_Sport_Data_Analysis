from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

from src.deep_models import build_deep_model
from src.wisdm_data import load_wisdm_cache
from train_wisdm_deep import make_loader, normalize_by_train, predict_all, run_epoch, subject_masks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the selected WISDM architecture on train+validation subjects and evaluate on test subjects."
    )
    parser.add_argument("--cache", default="models/wisdm_deep/fused_windows_phone.npz")
    parser.add_argument("--architecture", default="dual_branch_bigru")
    parser.add_argument("--model-out", default="models/wisdm_deep/wisdm_deep_model.pt")
    parser.add_argument("--report-dir", default="reports/wisdm_deep")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    bundle = load_wisdm_cache(Path(args.cache))
    train_mask, val_mask, test_mask = subject_masks(bundle.subjects)
    train_val_mask = train_mask | val_mask
    X, mean, std = normalize_by_train(bundle.X, train_val_mask)
    y = bundle.y

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_deep_model(args.architecture, n_channels=X.shape[1], n_classes=len(bundle.label_names)).to(device)

    train_loader = make_loader(X, y, train_val_mask, args.batch_size, shuffle=True)
    test_loader = make_loader(X, y, test_mask, args.batch_size, shuffle=False)

    counts = np.bincount(y[train_val_mask], minlength=len(bundle.label_names)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    print(
        f"Final training {args.architecture} on train+validation subjects. "
        f"train_val={train_val_mask.sum()}, test={test_mask.sum()}, device={device}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
        print(f"Epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

    y_true, y_pred = predict_all(model, test_loader, device)
    test_acc = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
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
            "test_macro_f1": float(test_macro_f1),
            "final_training": "train+validation subjects",
        },
        args.model_out,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wisdm_final_test_classification_report.txt").write_text(
        f"Model: {args.architecture}\n"
        "Training data: train+validation subjects after architecture selection\n"
        f"Test accuracy: {test_acc:.4f}\n"
        f"Test macro F1: {test_macro_f1:.4f}\n\n{report}",
        encoding="utf-8",
    )
    np.savetxt(report_dir / "wisdm_final_test_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    (report_dir / "wisdm_final_training.json").write_text(
        json.dumps(
            {
                "architecture": args.architecture,
                "epochs": args.epochs,
                "test_accuracy": test_acc,
                "test_macro_f1": test_macro_f1,
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved final model: {args.model_out}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(report)


if __name__ == "__main__":
    main()
