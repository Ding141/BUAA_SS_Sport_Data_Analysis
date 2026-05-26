"""Model testing interface for trained UCI HAR windows."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.config import ACTIVITY_NAMES, DEFAULT_DATA_ROOT, DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH
from src.data import apply_scaler, load_official_train_test, scaler_from_dict
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict UCI HAR activity from a sensor window.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--input", type=Path, default=None, help=".npy or .csv window file.")
    parser.add_argument("--sample-index", type=int, default=None, help="Use official test-set sample index.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def load_window(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".csv":
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    else:
        raise ValueError("Only .npy and .csv input windows are supported.")

    if arr.shape == (128, 9):
        arr = arr.T
    if arr.shape != (9, 128):
        raise ValueError(f"Expected window shape (9, 128) or (128, 9), got {arr.shape}.")
    return arr.astype(np.float32)


def load_model(model_path: Path, metadata: dict[str, object], device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(
        in_channels=int(checkpoint.get("in_channels", 9)),
        num_classes=int(checkpoint.get("num_classes", 6)),
        dropout=float(checkpoint.get("dropout", 0.25)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_window(model: torch.nn.Module, window: np.ndarray, metadata: dict[str, object], device: torch.device) -> dict[str, object]:
    scaler = scaler_from_dict(metadata["scaler"])
    x = apply_scaler(window[None, :, :], scaler)
    logits = model(torch.tensor(x, dtype=torch.float32, device=device))
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    labels = metadata.get("activity_names", ACTIVITY_NAMES)
    order = np.argsort(probs)[::-1]
    return {
        "predicted_index": int(order[0]),
        "predicted_label": labels[int(order[0])],
        "confidence": float(probs[order[0]]),
        "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
    }


def main() -> None:
    args = parse_args()
    if args.input is None and args.sample_index is None:
        args.sample_index = 0

    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, metadata, device)

    true_label = None
    if args.input is not None:
        window = load_window(args.input)
        source = str(args.input)
    else:
        data = load_official_train_test(args.data_root)
        idx = int(args.sample_index)
        window = data["x_test"][idx]
        true_label = metadata["activity_names"][int(data["y_test"][idx])]
        source = f"official test sample {idx}"

    result = predict_window(model, window, metadata, device)
    result["source"] = source
    if true_label is not None:
        result["true_label"] = true_label

    top_k = max(1, min(args.top_k, len(result["probabilities"])))
    ranked = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)[:top_k]

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\ntop predictions")
    writer = csv.writer(__import__("sys").stdout)
    writer.writerow(["rank", "label", "probability"])
    for rank, (label, prob) in enumerate(ranked, start=1):
        writer.writerow([rank, label, f"{prob:.6f}"])


if __name__ == "__main__":
    main()

