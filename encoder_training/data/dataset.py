"""WISDM raw data loading and windowing for self-supervised pretraining.

Each raw file: subject_id,activity_label,timestamp,x,y,z;
We use only watch data (accel + gyro), 6 channels total.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple


class WISDMWindowDataset(Dataset):
    """Load raw watch sensor data, split into overlapping windows, return (6, 200) tensors."""

    ACTIVITY_MAP = {
        'A': 'walking', 'B': 'jogging', 'C': 'stairs', 'D': 'sitting',
        'E': 'standing', 'F': 'typing', 'G': 'teeth', 'H': 'soup',
        'I': 'chips', 'J': 'pasta', 'K': 'drinking', 'L': 'sandwich',
        'M': 'kicking', 'O': 'catch', 'P': 'dribbling', 'Q': 'writing',
        'R': 'clapping', 'S': 'folding',
    }
    ACTIVITY_TO_IDX = {k: i for i, k in enumerate(ACTIVITY_MAP.keys())}
    N_CLASSES = 18

    def __init__(self, raw_dir: str, subject_ids: List[int], window_size: int = 200,
                 window_stride: int = 50, sensors=("accel", "gyro")):
        self.raw_dir = Path(raw_dir)
        self.subject_ids = sorted(subject_ids)
        self.window_size = window_size
        self.window_stride = window_stride
        self.sensors = list(sensors)
        self.n_channels = len(sensors) * 3  # each sensor has x, y, z

        self._windows: List[dict] = []
        self._build_index()

    def _parse_line(self, line: str) -> Tuple[int, str, float, float, float]:
        """Parse: subject_id,activity_label,timestamp,x,y,z;"""
        line = line.rstrip(';\n')
        parts = line.split(',')
        subject = int(parts[0])
        label = parts[1].strip()
        ts = float(parts[2])
        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
        return subject, label, ts, x, y, z

    def _build_index(self):
        for subj_id in self.subject_ids:
            # Load both accel and gyro for this subject
            sensor_data: Dict[str, np.ndarray] = {}
            sample_count = None

            for sensor in self.sensors:
                file_path = self.raw_dir / sensor / f"data_{subj_id}_{sensor}_watch.txt"
                if not file_path.exists():
                    break

                with open(file_path, 'r') as f:
                    lines = f.readlines()

                data = np.zeros((len(lines), 3), dtype=np.float32)
                labels = np.empty(len(lines), dtype='U1')
                for i, line in enumerate(lines):
                    _, label, _, x, y, z = self._parse_line(line)
                    data[i] = [x, y, z]
                    labels[i] = label
                sensor_data[sensor] = data
                if sample_count is None:
                    sample_count = len(lines)
                elif len(lines) != sample_count:
                    # Truncate to shorter if mismatch
                    sample_count = min(sample_count, len(lines))

            if len(sensor_data) != len(self.sensors):
                continue  # skip if any sensor missing

            # Concatenate channels: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
            channels = []
            for sensor in self.sensors:
                channels.append(sensor_data[sensor][:sample_count])
            full_signal = np.concatenate(channels, axis=1)  # (T, 6)

            # Take label from accel
            full_labels = labels[:sample_count]

            # Create sliding windows
            n_windows = max(0, (sample_count - self.window_size) // self.window_stride + 1)
            for w in range(n_windows):
                start = w * self.window_stride
                end = start + self.window_size
                window_data = full_signal[start:end]  # (200, 6)

                # Use majority label in window
                window_labels = full_labels[start:end]
                unique, counts = np.unique(window_labels, return_counts=True)
                majority_label = unique[np.argmax(counts)]

                self._windows.append({
                    'subject': subj_id,
                    'start': start,
                    'data': window_data,  # (200, 6)
                    'label': majority_label,
                    'label_idx': self.ACTIVITY_TO_IDX.get(majority_label, -1),
                })

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        win = self._windows[idx]
        # Return (6, 200) = channels first for Conv1d
        x = torch.from_numpy(win['data']).float().T  # (6, 200)
        return {
            'x': x,
            'subject': win['subject'],
            'label_idx': win['label_idx'],
            'label': win['label'],
        }
