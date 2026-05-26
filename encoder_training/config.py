from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    # Data paths
    dataset_root: str = "d:/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/wisdm-dataset"
    raw_dir: str = "raw/watch"  # watch data primary (hand fine-grained activities)
    sensors: List[str] = field(default_factory=lambda: ["accel", "gyro"])

    # Window parameters
    window_size: int = 200       # 10 seconds @ 20Hz
    window_stride: int = 50      # 2.5s stride, 75% overlap
    sampling_rate: int = 20

    # Train/val split by subjects
    train_subjects: List[int] = field(default_factory=lambda: list(range(1600, 1635)))
    val_subjects: List[int]   = field(default_factory=lambda: list(range(1635, 1643)))
    test_subjects: List[int]  = field(default_factory=lambda: list(range(1643, 1651)))


@dataclass
class MaskConfig:
    # Time-domain block masking (50% probability)
    mask_ratio: float = 0.4           # total fraction of patches to mask
    time_block_small: tuple = (3, 8)   # small blocks (60% of masked area)
    time_block_med: tuple   = (10, 18) # medium blocks (30% of masked area)
    time_block_large: tuple = (20, 30) # large blocks (10% of masked area)
    time_prob: float = 0.5             # probability of using time-only masking

    # Channel masking (25% probability)
    channel_mask_count: tuple = (1, 2) # mask 1-2 channels out of 6
    channel_prob: float = 0.25

    # Frequency masking (25% probability, uses STFT)
    freq_mask_ratio: float = 0.3       # fraction of frequency bins to mask
    freq_mask_bias_high: float = 0.7   # 70% of masked bins from mid-high frequencies
    freq_prob: float = 0.25

    # STFT params
    stft_n_fft: int = 32
    stft_hop: int = 8


@dataclass
class EncoderConfig:
    # Patch embedding (Conv Stem)
    patch_size: int = 10
    n_patches: int = 20                # 200 / 10
    n_channels: int = 6                # accel_x,y,z + gyro_x,y,z
    embed_dim: int = 256
    conv_kernel_size: int = 10         # same as patch_size for non-overlapping conv

    # Global frequency token
    use_global_freq_token: bool = True
    global_freq_fft_points: int = 200
    global_freq_mlp_hidden: int = 128

    # Transformer encoder (shared between x and y)
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Predictor (lightweight decoder)
    predictor_n_layers: int = 2
    predictor_mlp_hidden: int = 512

    # y-encoder targets
    target_layers: List[int] = field(default_factory=lambda: [3, 6])  # layer indices for multi-level targets
    target_layer_weights: List[float] = field(default_factory=lambda: [0.3, 1.0])


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    grad_clip: float = 1.0

    # EMA for y-encoder
    ema_tau_start: float = 0.999
    ema_tau_end: float = 0.9999

    # SigReg weights
    lambda_var: float = 0.1
    lambda_ent: float = 0.01

    # Logging
    log_interval: int = 10
    save_dir: str = "d:/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/encoder_training/checkpoints"

    # Device
    device: str = "cuda"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
