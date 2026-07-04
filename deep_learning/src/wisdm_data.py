from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


WISDM_SAMPLE_RATE_HZ = 20.0
DEFAULT_WINDOW_SIZE = 200
DEFAULT_STRIDE = 100

ACTIVITY_NAME_BY_CODE = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "teeth",
    "H": "soup",
    "I": "chips",
    "J": "pasta",
    "K": "drinking",
    "L": "sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding",
}


@dataclass(frozen=True)
class WisdmBundle:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    label_codes: list[str]
    label_names: list[str]
    sample_rate_hz: float
    window_size: int
    stride: int
    channels: list[str]
    device: str


def _read_sensor_file(path: Path) -> dict[str, np.ndarray]:
    grouped: dict[str, list[list[float]]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 6:
                continue
            label = parts[1]
            try:
                xyz = [float(parts[3]), float(parts[4]), float(parts[5])]
            except ValueError:
                continue
            grouped.setdefault(label, []).append(xyz)
    return {label: np.asarray(values, dtype=np.float32) for label, values in grouped.items()}


def load_wisdm_fused_windows(
    data_dir: Path,
    device: str = "phone",
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    max_windows_per_class: int | None = None,
) -> WisdmBundle:
    """Create accel+gyro WISDM windows.

    Windows are paired by subject and activity. For each subject/activity stream,
    accelerometer and gyroscope samples are truncated to the shorter length and
    stacked into a 6-channel time series: accel_xyz + gyro_xyz.
    """
    if device not in {"phone", "watch"}:
        raise ValueError("device must be 'phone' or 'watch'")

    raw_dir = data_dir / "raw" / device
    accel_dir = raw_dir / "accel"
    gyro_dir = raw_dir / "gyro"
    label_codes = sorted(ACTIVITY_NAME_BY_CODE)
    code_to_id = {code: i for i, code in enumerate(label_codes)}
    X_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    subject_parts: list[int] = []

    for accel_path in sorted(accel_dir.glob(f"data_*_accel_{device}.txt")):
        subject = int(accel_path.name.split("_")[1])
        gyro_path = gyro_dir / f"data_{subject}_gyro_{device}.txt"
        if not gyro_path.exists():
            continue

        accel_by_label = _read_sensor_file(accel_path)
        gyro_by_label = _read_sensor_file(gyro_path)

        for code in label_codes:
            if code not in accel_by_label or code not in gyro_by_label:
                continue
            accel = accel_by_label[code]
            gyro = gyro_by_label[code]
            n = min(len(accel), len(gyro))
            if n < window_size:
                continue
            fused = np.concatenate([accel[:n], gyro[:n]], axis=1)

            for start in range(0, n - window_size + 1, stride):
                window = fused[start : start + window_size].T
                X_parts.append(window.astype(np.float32))
                y_parts.append(code_to_id[code])
                subject_parts.append(subject)

    if not X_parts:
        raise RuntimeError(f"No WISDM windows were created from {raw_dir}")

    X = np.stack(X_parts, axis=0)
    y = np.asarray(y_parts, dtype=np.int64)
    subjects = np.asarray(subject_parts, dtype=np.int64)
    if max_windows_per_class is not None:
        rng = np.random.default_rng(42)
        keep: list[np.ndarray] = []
        for class_id in range(len(label_codes)):
            indices = np.flatnonzero(y == class_id)
            if len(indices) > max_windows_per_class:
                indices = rng.choice(indices, size=max_windows_per_class, replace=False)
            keep.append(indices)
        selected = np.sort(np.concatenate(keep))
        X = X[selected]
        y = y[selected]
        subjects = subjects[selected]

    return WisdmBundle(
        X=X,
        y=y,
        subjects=subjects,
        label_codes=label_codes,
        label_names=[ACTIVITY_NAME_BY_CODE[code] for code in label_codes],
        sample_rate_hz=WISDM_SAMPLE_RATE_HZ,
        window_size=window_size,
        stride=stride,
        channels=["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
        device=device,
    )


def save_wisdm_cache(bundle: WisdmBundle, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=bundle.X,
        y=bundle.y,
        subjects=bundle.subjects,
        label_codes=np.asarray(bundle.label_codes),
        label_names=np.asarray(bundle.label_names),
        sample_rate_hz=np.asarray(bundle.sample_rate_hz),
        window_size=np.asarray(bundle.window_size),
        stride=np.asarray(bundle.stride),
        channels=np.asarray(bundle.channels),
        device=np.asarray(bundle.device),
    )


def load_wisdm_cache(path: Path) -> WisdmBundle:
    data = np.load(path, allow_pickle=False)
    return WisdmBundle(
        X=data["X"],
        y=data["y"],
        subjects=data["subjects"],
        label_codes=[str(x) for x in data["label_codes"].tolist()],
        label_names=[str(x) for x in data["label_names"].tolist()],
        sample_rate_hz=float(data["sample_rate_hz"]),
        window_size=int(data["window_size"]),
        stride=int(data["stride"]),
        channels=[str(x) for x in data["channels"].tolist()],
        device=str(data["device"]),
    )


def load_fused_window_file(path: Path, window_size: int = DEFAULT_WINDOW_SIZE) -> np.ndarray:
    """Load one accel+gyro window from txt/csv/npy.

    Supported shapes are 6 x window_size, window_size x 6, or a flattened vector.
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        try:
            arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        except ValueError:
            arr = np.loadtxt(path, dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size != 6 * window_size:
            raise ValueError(f"1-D input must contain exactly {6 * window_size} values")
        arr = arr.reshape(6, window_size)
    elif arr.shape == (window_size, 6):
        arr = arr.T
    elif arr.shape != (6, window_size):
        raise ValueError(f"Input window must have shape 6x{window_size} or {window_size}x6")
    return arr
