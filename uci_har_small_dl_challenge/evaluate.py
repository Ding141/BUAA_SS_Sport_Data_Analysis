"""Evaluate a trained model on the official UCI HAR test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import ACTIVITY_NAMES, DEFAULT_DATA_ROOT, DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, REPORT_DIR
from src.data import apply_scaler, load_official_train_test, scaler_from_dict
from src.model import build_model
from src.train_utils import evaluate_predictions, make_loader, predict_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained UCI HAR model.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model = build_model(
        in_channels=int(checkpoint.get("in_channels", 9)),
        num_classes=int(checkpoint.get("num_classes", 6)),
        dropout=float(checkpoint.get("dropout", 0.25)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data = load_official_train_test(args.data_root)
    scaler = scaler_from_dict(metadata["scaler"])
    x_test = apply_scaler(data["x_test"], scaler)
    loader = make_loader(x_test, data["y_test"], args.batch_size, shuffle=False)
    logits, labels = predict_logits(model, loader, device)
    preds = logits.argmax(axis=1)
    metrics = evaluate_predictions(labels, preds, metadata.get("activity_names", ACTIVITY_NAMES))
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"accuracy:    {metrics['accuracy']:.4f}")
    print(f"macro_f1:    {metrics['macro_f1']:.4f}")
    print("confusion matrix:")
    print(np.asarray(metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()

