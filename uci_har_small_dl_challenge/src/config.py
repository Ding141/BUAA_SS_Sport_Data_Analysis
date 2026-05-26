"""Project configuration and label metadata."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "uci" / "UCI HAR Dataset"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
EXAMPLE_DIR = PROJECT_ROOT / "examples"

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

CHANNEL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

DEFAULT_MODEL_PATH = MODEL_DIR / "uci_har_cnn_attention.pt"
DEFAULT_METADATA_PATH = MODEL_DIR / "uci_har_cnn_attention_meta.json"
