from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.wisdm_data import load_fused_window_file
from src.deep_models import build_deep_model


def possible_fall_warning(window: np.ndarray) -> tuple[bool, float]:
    """Unsupervised warning only; WISDM has no fall class for supervised training."""
    accel_mag = np.linalg.norm(window[:3], axis=0)
    peak_g = float(np.max(accel_mag) / 9.80665)
    return peak_g >= 2.5, peak_g


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct WISDM deep-model test entry.")
    parser.add_argument("--model", default="models/wisdm_deep/wisdm_deep_model.pt", help="Deep model checkpoint")
    parser.add_argument("--input", help="A txt/csv/npy file containing one 6x200 or 200x6 window")
    parser.add_argument("--cache", default="models/wisdm_deep/fused_windows_phone.npz", help="WISDM fused-window cache")
    parser.add_argument("--sample", type=int, default=0, help="Sample index used when --input is omitted")
    parser.add_argument("--top-k", type=int, default=3, help="Number of probabilities to print")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    window_size = int(checkpoint["window_size"])
    true_label = None
    if args.input:
        window = load_fused_window_file(Path(args.input), window_size=window_size)
        source = args.input
    else:
        from src.wisdm_data import load_wisdm_cache

        cache = load_wisdm_cache(Path(args.cache))
        window = cache.X[args.sample]
        true_label = cache.label_names[int(cache.y[args.sample])]
        source = f"{args.cache} sample #{args.sample}"

    x = (window[np.newaxis, :, :] - checkpoint["mean"]) / checkpoint["std"]
    x_tensor = torch.from_numpy(x.astype(np.float32))

    model = build_deep_model(
        checkpoint.get("model_type", "cnn_bigru"),
        n_channels=x_tensor.shape[1],
        n_classes=len(checkpoint["label_names"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        probabilities = torch.softmax(model(x_tensor), dim=1).numpy()[0]

    order = np.argsort(probabilities)[::-1]
    best = int(order[0])
    print(f"Input: {source}")
    if true_label is not None:
        print(f"True activity: {true_label}")
    print(f"Predicted label code: {checkpoint['label_codes'][best]}")
    print(f"Predicted activity: {checkpoint['label_names'][best]}")
    print(f"Confidence: {probabilities[best]:.4f}")
    print("Top classes:")
    for idx in order[: args.top_k]:
        print(f"  {checkpoint['label_codes'][int(idx)]}: {checkpoint['label_names'][int(idx)]} {probabilities[int(idx)]:.4f}")

    warning, peak_g = possible_fall_warning(window)
    print(f"Peak acceleration: {peak_g:.2f} g")
    print(f"Possible fall warning: {'YES' if warning else 'NO'}")
    if warning:
        print("Note: this is a threshold warning, not a supervised fall class; WISDM has no fall label.")


if __name__ == "__main__":
    main()
