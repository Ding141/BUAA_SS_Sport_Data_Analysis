"""Train the UCI HAR small-class deep learning challenge model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    ACTIVITY_NAMES,
    CHANNEL_NAMES,
    DEFAULT_DATA_ROOT,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    EXAMPLE_DIR,
    MODEL_DIR,
    REPORT_DIR,
)
from src.data import (
    apply_scaler,
    fit_scaler,
    load_official_train_test,
    save_example_windows,
    scaler_to_dict,
)
from src.model import build_model
from src.train_utils import (
    evaluate_predictions,
    make_loader,
    predict_logits,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNNResidualAttention on UCI HAR.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.18)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"data root: {args.data_root}")

    data = load_official_train_test(args.data_root)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(
        splitter.split(data["x_train"], data["y_train"], groups=data["subjects_train"])
    )

    x_fit = data["x_train"][train_idx]
    scaler = fit_scaler(x_fit)
    x_train = apply_scaler(data["x_train"][train_idx], scaler)
    y_train = data["y_train"][train_idx]
    x_val = apply_scaler(data["x_train"][val_idx], scaler)
    y_val = data["y_train"][val_idx]
    x_test = apply_scaler(data["x_test"], scaler)
    y_test = data["y_test"]

    print(f"train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}")
    print(f"train subjects: {sorted(np.unique(data['subjects_train'][train_idx]).tolist())}")
    print(f"val subjects: {sorted(np.unique(data['subjects_train'][val_idx]).tolist())}")

    model = build_model(
        in_channels=x_train.shape[1],
        num_classes=len(ACTIVITY_NAMES),
        dropout=args.dropout,
    ).to(device)

    result = train_model(
        model=model,
        train_loader=make_loader(x_train, y_train, args.batch_size, shuffle=True),
        val_loader=make_loader(x_val, y_val, args.batch_size * 2, shuffle=False),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        patience=args.patience,
    )

    model.load_state_dict(result.best_state)
    test_loader = make_loader(x_test, y_test, args.batch_size * 2, shuffle=False)
    logits, labels = predict_logits(model, test_loader, device)
    preds = logits.argmax(axis=1)
    metrics = evaluate_predictions(labels, preds, ACTIVITY_NAMES)

    checkpoint = {
        "state_dict": result.best_state,
        "model_name": "CNNResidualAttention",
        "in_channels": int(x_train.shape[1]),
        "num_classes": len(ACTIVITY_NAMES),
        "dropout": args.dropout,
    }
    torch.save(checkpoint, args.model_path)

    metadata = {
        "dataset": "UCI HAR Dataset",
        "task": "six-class human activity recognition with multi-sensor fusion",
        "model_name": "CNNResidualAttention",
        "activity_names": ACTIVITY_NAMES,
        "channel_names": CHANNEL_NAMES,
        "input_shape": [9, 128],
        "scaler": scaler_to_dict(scaler),
        "best_val_accuracy": result.best_val_accuracy,
        "test_metrics": metrics,
        "train_args": vars(args) | {
            "data_root": str(args.data_root),
            "model_path": str(args.model_path),
            "metadata_path": str(args.metadata_path),
        },
    }
    args.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPORT_DIR / "training_history.json").write_text(
        json.dumps(result.history, indent=2),
        encoding="utf-8",
    )
    (REPORT_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    save_example_windows(EXAMPLE_DIR, args.data_root, max_per_class=1)

    print("\nfinal official test metrics")
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"macro_f1:  {metrics['macro_f1']:.4f}")
    print(f"model:     {args.model_path}")
    print(f"metadata:  {args.metadata_path}")


if __name__ == "__main__":
    main()

