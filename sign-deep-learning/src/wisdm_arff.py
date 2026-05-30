from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .wisdm_data import ACTIVITY_NAME_BY_CODE


@dataclass(frozen=True)
class WisdmArffBundle:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    label_codes: list[str]
    label_names: list[str]
    feature_names: list[str]
    source: str


def _read_arff(path: Path, prefix: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    attrs: list[str] = []
    labels: list[str] = []
    subjects: list[int] = []
    rows: list[list[float]] = []
    in_data = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            lower = s.lower()
            if lower.startswith("@attribute"):
                name = s.split(maxsplit=2)[1].strip('"')
                attrs.append(name)
            elif lower.startswith("@data"):
                in_data = True
            elif in_data:
                parts = [p.strip() for p in s.split(",")]
                if len(parts) < 3:
                    continue
                labels.append(parts[0])
                subjects.append(int(float(parts[-1])))
                rows.append([float(v) for v in parts[1:-1]])

    feature_names = [f"{prefix}_{name}" for name in attrs[1:-1]]
    return feature_names, np.asarray(rows, dtype=np.float32), np.asarray(labels), np.asarray(subjects, dtype=np.int64)


def load_wisdm_arff_fused(data_dir: Path, device: str = "phone") -> WisdmArffBundle:
    arff_root = data_dir / "arff_files" / device
    label_codes = sorted(ACTIVITY_NAME_BY_CODE)
    code_to_id = {code: i for i, code in enumerate(label_codes)}

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for accel_path in sorted((arff_root / "accel").glob(f"data_*_accel_{device}.arff")):
        subject = accel_path.name.split("_")[1]
        gyro_path = arff_root / "gyro" / f"data_{subject}_gyro_{device}.arff"
        if not gyro_path.exists():
            continue

        accel_names, accel_X, accel_labels, accel_subjects = _read_arff(accel_path, "accel")
        gyro_names, gyro_X, gyro_labels, gyro_subjects = _read_arff(gyro_path, "gyro")
        n = min(len(accel_X), len(gyro_X))
        if n == 0:
            continue

        labels = accel_labels[:n]
        same_label = labels == gyro_labels[:n]
        same_subject = accel_subjects[:n] == gyro_subjects[:n]
        keep = same_label & same_subject
        if not np.any(keep):
            continue

        # Filter to only include labels present in ACTIVITY_NAME_BY_CODE
        valid_mask = np.asarray([str(code) in code_to_id for code in labels[keep]], dtype=bool)
        if not np.any(valid_mask):
            continue
        keep_indices = np.flatnonzero(keep)
        keep_indices = keep_indices[valid_mask]
        labels_filtered = labels[keep_indices]

        fused = np.concatenate([accel_X[:n][keep_indices], gyro_X[:n][keep_indices]], axis=1)
        y = np.asarray([code_to_id[str(code)] for code in labels_filtered], dtype=np.int64)
        subjects = accel_subjects[:n][keep_indices]
        X_parts.append(fused)
        y_parts.append(y)
        subject_parts.append(subjects)
        if feature_names is None:
            feature_names = accel_names + gyro_names

    if not X_parts:
        raise RuntimeError(f"No ARFF features found under {arff_root}")

    return WisdmArffBundle(
        X=np.concatenate(X_parts, axis=0),
        y=np.concatenate(y_parts, axis=0),
        subjects=np.concatenate(subject_parts, axis=0),
        label_codes=label_codes,
        label_names=[ACTIVITY_NAME_BY_CODE[code] for code in label_codes],
        feature_names=feature_names or [],
        source=f"{device} accel+gyro ARFF",
    )


def save_wisdm_arff_cache(bundle: WisdmArffBundle, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=bundle.X,
        y=bundle.y,
        subjects=bundle.subjects,
        label_codes=np.asarray(bundle.label_codes),
        label_names=np.asarray(bundle.label_names),
        feature_names=np.asarray(bundle.feature_names),
        source=np.asarray(bundle.source),
    )


def load_wisdm_arff_cache(path: Path) -> WisdmArffBundle:
    data = np.load(path, allow_pickle=False)
    return WisdmArffBundle(
        X=data["X"].astype(np.float32),
        y=data["y"].astype(np.int64),
        subjects=data["subjects"].astype(np.int64),
        label_codes=[str(x) for x in data["label_codes"].tolist()],
        label_names=[str(x) for x in data["label_names"].tolist()],
        feature_names=[str(x) for x in data["feature_names"].tolist()],
        source=str(data["source"]),
    )
