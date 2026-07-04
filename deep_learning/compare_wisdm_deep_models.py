from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.deep_models import build_deep_model
from src.wisdm_data import load_wisdm_cache, load_wisdm_fused_windows, save_wisdm_cache
from train_wisdm_deep import (
    make_loader,
    normalize_by_train,
    predict_all,
    run_epoch,
    subject_masks,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train_candidate(
    architecture: str,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    label_names: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(42)
    np.random.seed(42)
    model = build_deep_model(architecture, n_channels=X.shape[1], n_classes=len(label_names)).to(device)

    train_loader = make_loader(X, y, train_mask, batch_size, shuffle=True)
    val_loader = make_loader(X, y, val_mask, batch_size, shuffle=False)
    test_loader = make_loader(X, y, test_mask, batch_size, shuffle=False)

    counts = np.bincount(y[train_mask], minlength=len(label_names)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = -1.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"{architecture} epoch {epoch:02d}/{epochs} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    val_true, val_pred = predict_all(model, val_loader, device)
    test_true, test_pred = predict_all(model, test_loader, device)

    return {
        "architecture": architecture,
        "model": model,
        "state": model.state_dict(),
        "history": history,
        "val_accuracy": accuracy_score(val_true, val_pred),
        "val_macro_f1": f1_score(val_true, val_pred, average="macro", zero_division=0),
        "test_accuracy": accuracy_score(test_true, test_pred),
        "test_macro_f1": f1_score(test_true, test_pred, average="macro", zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deep architectures on WISDM accel+gyro windows.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "wisdm-dataset"))
    parser.add_argument("--device-type", choices=["phone", "watch"], default="phone")
    parser.add_argument("--cache", default="models/wisdm_deep/fused_windows_phone.npz")
    parser.add_argument("--report-dir", default="reports/wisdm_deep")
    parser.add_argument("--model-dir", default="models/wisdm_deep")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["cnn1d", "cnn_gru", "cnn_bigru", "cnn_lstm", "cnn_bilstm", "transformer", "dual_branch_bigru"],
    )
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if cache_path.exists():
        bundle = load_wisdm_cache(cache_path)
    else:
        bundle = load_wisdm_fused_windows(Path(args.data_dir), device=args.device_type)
        save_wisdm_cache(bundle, cache_path)

    train_mask, val_mask, test_mask = subject_masks(bundle.subjects)
    X, mean, std = normalize_by_train(bundle.X, train_mask)
    y = bundle.y
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparing on {device}. Windows: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    report_dir = Path(args.report_dir)
    model_dir = Path(args.model_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_result = None
    for architecture in args.architectures:
        print(f"\n=== {architecture} ===")
        result = train_candidate(
            architecture,
            X,
            y,
            train_mask,
            val_mask,
            test_mask,
            bundle.label_names,
            args.epochs,
            args.batch_size,
            args.lr,
            device,
        )
        rows.append(
            {
                "architecture": architecture,
                "val_accuracy": result["val_accuracy"],
                "val_macro_f1": result["val_macro_f1"],
                "test_accuracy": result["test_accuracy"],
                "test_macro_f1": result["test_macro_f1"],
            }
        )
        candidate_path = model_dir / f"{architecture}.pt"
        torch.save(
            {
                "model_state": result["state"],
                "model_type": architecture,
                "label_codes": bundle.label_codes,
                "label_names": bundle.label_names,
                "channels": bundle.channels,
                "window_size": bundle.window_size,
                "sample_rate_hz": bundle.sample_rate_hz,
                "mean": mean,
                "std": std,
                "device_source": bundle.device,
                "test_accuracy": float(result["test_accuracy"]),
                "test_macro_f1": float(result["test_macro_f1"]),
                "val_accuracy": float(result["val_accuracy"]),
                "val_macro_f1": float(result["val_macro_f1"]),
            },
            candidate_path,
        )
        print(
            f"{architecture}: val_acc={result['val_accuracy']:.4f}, "
            f"test_acc={result['test_accuracy']:.4f}"
        )
        if best_result is None or result["val_accuracy"] > best_result["val_accuracy"]:
            best_result = {**result, "path": candidate_path}

    csv_path = report_dir / "deep_model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["architecture", "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"],
        )
        writer.writeheader()
        writer.writerows(rows)

    assert best_result is not None
    final_path = model_dir / "wisdm_deep_model.pt"
    shutil.copyfile(best_result["path"], final_path)
    (report_dir / "best_deep_model.json").write_text(
        json.dumps(
            {
                "selected_by": "highest validation accuracy on held-out subjects",
                "best_architecture": best_result["architecture"],
                "val_accuracy": best_result["val_accuracy"],
                "val_macro_f1": best_result["val_macro_f1"],
                "test_accuracy": best_result["test_accuracy"],
                "test_macro_f1": best_result["test_macro_f1"],
                "comparison": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nBest architecture: {best_result['architecture']}")
    print(f"Saved comparison: {csv_path}")
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
