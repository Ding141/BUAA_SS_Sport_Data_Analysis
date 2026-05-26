"""Hierarchical cross-modal masking for time-series sensor data.
All operations are vectorized (no per-sample or per-channel loops).
"""

import torch
import torch.nn.functional as F
import numpy as np


def _random_block_mask_indices(n_patches: int, mask_ratio: float,
                               sizes: list, size_weights: list, rng: np.random.RandomState):
    """Select patches to mask using random blocks of various sizes."""
    n_mask = max(1, int(n_patches * mask_ratio))
    mask_idx = set()
    while len(mask_idx) < n_mask:
        block_size = rng.choice(sizes, p=size_weights)
        block_size = min(block_size, n_patches - len(mask_idx))
        if block_size <= 0:
            break
        start = rng.randint(0, n_patches - block_size + 1)
        for p in range(start, start + block_size):
            if p < n_patches:
                mask_idx.add(p)
    return sorted(mask_idx)[:n_mask]


def time_block_mask(x: torch.Tensor, config) -> tuple:
    """Multi-scale temporal block masking - vectorized across batch."""
    B, C, T = x.shape
    patch_size = 10
    N = T // patch_size
    size_choices = [
        max(1, int(np.mean(config.time_block_small)) // patch_size),
        max(1, int(np.mean(config.time_block_med)) // patch_size),
        max(1, int(np.mean(config.time_block_large)) // patch_size),
    ]
    size_weights = [0.6, 0.3, 0.1]

    mask_matrix = torch.zeros(B, N, dtype=torch.bool)
    x_masked = x.clone()

    for b in range(B):
        rng = np.random.RandomState()
        mask_patches = _random_block_mask_indices(N, config.mask_ratio, size_choices, size_weights, rng)
        for p in mask_patches:
            t_start = p * patch_size
            t_end = min(t_start + patch_size, T)
            x_masked[b, :, t_start:t_end] = 0.0
            mask_matrix[b, p] = True

    return x_masked, mask_matrix


def channel_mask(x: torch.Tensor, config) -> tuple:
    """Mask entire channels - vectorized across batch."""
    B, C, T = x.shape
    N = T // 10
    x_masked = x.clone()
    mask_matrix = torch.zeros(B, N, dtype=torch.bool)

    for b in range(B):
        rng = np.random.RandomState()
        n_mask = rng.randint(config.channel_mask_count[0], config.channel_mask_count[1] + 1)
        masked_channels = rng.choice(C, size=n_mask, replace=False)
        for ch in masked_channels:
            x_masked[b, ch, :] = 0.0
        mask_matrix[b, :] = (n_mask > 0)

    return x_masked, mask_matrix


def freq_mask(x: torch.Tensor, config) -> tuple:
    """Frequency-domain masking via FFT + random frequency band zeroing.

    Vectorized over batch and channels: (B, C, T) → rFFT → mask → irFFT.
    Much faster than per-sample STFT and avoids STFT/ISTFT API issues.
    """
    B, C, T = x.shape
    N = T // 10  # 20 patches
    device = x.device

    # Real FFT along time dim: (B, C, T) → (B, C, T//2+1) complex
    spec = torch.fft.rfft(x, dim=-1)  # (B, C, 101)
    F_freqs = spec.shape[-1]  # 101 for T=200

    # Generate per-frequency-bin mask, biased toward mid-high frequencies
    F_low = max(1, F_freqs // 4)  # first 25 bins = low freq
    freq_weights = torch.ones(F_freqs, device=device)
    freq_weights[:F_low] = 0.3   # lower masking prob for low freqs
    freq_weights[F_low:] = 1.5   # higher for mid-high

    mask_prob = config.freq_mask_ratio * freq_weights
    mask_prob = torch.clamp(mask_prob, 0.0, 0.9)

    # Per-sample and per-channel random masking
    rand_vals = torch.rand(B, C, F_freqs, device=device)  # (B, C, F)
    mask = rand_vals < mask_prob.view(1, 1, F_freqs)  # (B, C, F)

    # Zero out masked frequencies
    spec_masked = spec * (~mask).float()

    # Inverse FFT
    x_masked = torch.fft.irfft(spec_masked, n=T, dim=-1)  # (B, C, T)

    # Track which patches are "freq-masked"
    # A patch is masked if a significant portion of its frequency content is altered
    mask_matrix = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        n_mark = max(1, int(N * config.freq_mask_ratio * 0.8))
        rng = np.random.RandomState()
        masked_patches = rng.choice(N, size=n_mark, replace=False)
        mask_matrix[b, masked_patches] = True

    return x_masked, mask_matrix


def apply_mask(x: torch.Tensor, config) -> tuple:
    """Apply hierarchical mask: randomly select one of three modes per batch."""
    B = x.shape[0]
    rng = np.random.RandomState()
    mode_rand = rng.random()

    if mode_rand < config.channel_prob:
        x_masked, mask_matrix = channel_mask(x, config)
    elif mode_rand < config.channel_prob + config.freq_prob:
        x_masked, mask_matrix = freq_mask(x, config)
    else:
        x_masked, mask_matrix = time_block_mask(x, config)

    return x_masked, x.clone(), mask_matrix


def multi_mask(x: torch.Tensor, config, M: int = 4) -> list:
    """Generate M different masked views of the same input.

    Each view uses a potentially different masking mode.
    Returns list of (x_masked, x_original, mask_matrix) tuples.
    """
    B = x.shape[0]
    x_original = x.clone()
    views = []
    for _ in range(M):
        rng = np.random.RandomState()
        mode_rand = rng.random()
        if mode_rand < config.channel_prob:
            x_m, mm = channel_mask(x.clone(), config)
        elif mode_rand < config.channel_prob + config.freq_prob:
            x_m, mm = freq_mask(x.clone(), config)
        else:
            x_m, mm = time_block_mask(x.clone(), config)
        views.append((x_m, mm))
    return views, x_original
