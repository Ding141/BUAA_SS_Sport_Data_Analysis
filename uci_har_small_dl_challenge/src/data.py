"""Data loading and preprocessing for the UCI HAR small-class challenge."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import ACTIVITY_NAMES, CHANNEL_NAMES, DEFAULT_DATA_ROOT


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"UCI HAR data directory not found: {root}. "
            "Pass --data-root or set it in your command."
        )
    return root


def load_split(data_root: str | Path | None, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one official UCI HAR split.

    Returns:
        X: (n_samples, 9, 128), y: zero-based labels, subjects.
    """
    root = resolve_data_root(data_root)
    signal_dir = root / split / "Inertial Signals"
    channels = []

    for name in CHANNEL_NAMES:
        path = signal_dir / f"{name}_{split}.txt"
        channels.append(pd.read_csv(path, sep=r"\s+", header=None).to_numpy(np.float32))

    x = np.stack(channels, axis=1)
    y_path = root / split / f"y_{split}.txt"
    subject_path = root / split / f"subject_{split}.txt"
    y = pd.read_csv(y_path, sep=r"\s+", header=None).to_numpy().ravel().astype(np.int64) - 1
    subjects = pd.read_csv(subject_path, sep=r"\s+", header=None).to_numpy().ravel().astype(np.int64)
    return x, y, subjects


def load_official_train_test(data_root: str | Path | None = None) -> dict[str, np.ndarray]:
    x_train, y_train, s_train = load_split(data_root, "train")
    x_test, y_test, s_test = load_split(data_root, "test")
    return {
        "x_train": x_train,
        "y_train": y_train,
        "subjects_train": s_train,
        "x_test": x_test,
        "y_test": y_test,
        "subjects_test": s_test,
    }


def fit_scaler(x_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(x_train.transpose(0, 2, 1).reshape(-1, x_train.shape[1]))
    return scaler


def apply_scaler(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, c, t = x.shape
    flat = x.transpose(0, 2, 1).reshape(-1, c)
    scaled = scaler.transform(flat).reshape(n, t, c).transpose(0, 2, 1)
    return scaled.astype(np.float32)


def scaler_to_dict(scaler: StandardScaler) -> dict[str, list[float]]:
    return {
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
    }


def scaler_from_dict(payload: dict[str, list[float]]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(payload["scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def save_example_windows(
    output_dir: str | Path,
    data_root: str | Path | None = None,
    max_per_class: int = 1,
) -> None:
    """Save small prediction examples as .npy and metadata json files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_official_train_test(data_root)
    x = data["x_test"]
    y = data["y_test"]
    written: dict[int, int] = {}

    for idx, label in enumerate(y):
        count = written.get(int(label), 0)
        if count >= max_per_class:
            continue
        stem = f"{int(label)}_{ACTIVITY_NAMES[int(label)].lower()}_{count}"
        np.save(output / f"{stem}.npy", x[idx])
        (output / f"{stem}.json").write_text(
            json.dumps(
                {
                    "label_index": int(label),
                    "label_name": ACTIVITY_NAMES[int(label)],
                    "shape": list(x[idx].shape),
                    "channels": CHANNEL_NAMES,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written[int(label)] = count + 1
        if len(written) == len(ACTIVITY_NAMES) and all(v >= max_per_class for v in written.values()):
            break

