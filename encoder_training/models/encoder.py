"""Patch embedding and Transformer encoder. x-encoder now includes [MASK] tokens
at masked positions, enabling self-attention between visible and masked regions."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTimeFreqEmbedding(nn.Module):
    """Conv-stem per-patch time + local frequency fusion → embed_dim tokens."""

    def __init__(self, n_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        half_dim = embed_dim // 2
        self.conv_stem = nn.Conv1d(n_channels, half_dim, kernel_size=patch_size,
                                    stride=patch_size, bias=False)
        n_freq_bins = patch_size // 2 + 1
        self.freq_proj = nn.Linear(n_channels * n_freq_bins, half_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 21, embed_dim) * 0.02)
        self.fusion = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size

        # Global freq embedding (full-signal FFT → embed_dim)
        self.global_freq_proj = nn.Linear(101, embed_dim, bias=False)

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape
        # Time branch
        time_feat = self.conv_stem(x)  # (B, half_dim, 20)

        # Freq branch: per-patch local FFT
        x_patch = x.view(B, C, 20, self.patch_size)
        x_patch_centered = x_patch - x_patch.mean(dim=-1, keepdim=True)
        freq_mag = torch.abs(torch.fft.rfft(x_patch_centered, dim=-1))  # (B,C,20,6)
        freq_flat = freq_mag.permute(0, 2, 1, 3).reshape(B, 20, -1)  # (B,20,C*6)
        freq_feat = self.freq_proj(freq_flat).transpose(1, 2)  # (B, half_dim, 20)

        combined = torch.cat([time_feat, freq_feat], dim=1)  # (B, embed_dim, 20)
        tokens = self.fusion(combined.transpose(1, 2))  # (B, 20, embed_dim)

        # Global frequency token
        full_freq = torch.abs(torch.fft.rfft(x, dim=-1)).mean(dim=1)  # (B, 101)
        global_freq = self.global_freq_proj(full_freq).unsqueeze(1)  # (B, 1, dim)

        tokens = tokens + self.pos_embed[:, :20, :]
        global_freq = global_freq + self.pos_embed[:, 20:21, :]
        tokens = self.norm(tokens)
        global_freq = self.norm(global_freq)
        return tokens, global_freq


class TransformerEncoder(nn.Module):
    """Standard Transformer encoder, collects intermediate layer outputs."""

    def __init__(self, embed_dim: int, n_layers: int, n_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 collect_layers: list = None):
        super().__init__()
        self.n_layers = n_layers
        self.collect_layers = collect_layers or []
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                dim_feedforward=int(embed_dim*mlp_ratio), dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, self.dropout(x)], dim=1)

        intermediates = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            layer_idx = i + 1
            if layer_idx in self.collect_layers:
                intermediates[layer_idx] = self.final_norm(x[:, 1:, :])

        x = self.final_norm(x)
        cls_out = x[:, 0, :]
        patch_out = x[:, 1:, :]
        if self.n_layers in self.collect_layers:
            intermediates[self.n_layers] = patch_out
        return patch_out, intermediates, cls_out


class MaskedEncoder(nn.Module):
    """Encoder with two modes:
    - x-encoder mode (mask_matrix given): insert [MASK] tokens at masked positions,
      full self-attention over all 20 patches + global_freq.
    - y-encoder mode (mask_matrix=None): full original signal, no masking.
    """

    def __init__(self, config):
        super().__init__()
        self.embed = PatchTimeFreqEmbedding(
            n_channels=config.n_channels, patch_size=config.patch_size,
            embed_dim=config.embed_dim)
        self.transformer = TransformerEncoder(
            embed_dim=config.embed_dim, n_layers=config.n_layers,
            n_heads=config.n_heads, mlp_ratio=config.mlp_ratio,
            dropout=config.dropout, collect_layers=config.target_layers)
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        self.embed_dim = config.embed_dim

    def forward(self, x: torch.Tensor, mask_matrix=None):
        """
        Returns: patch_out (B, N_out, dim), intermediates dict, cls_out (B, dim),
                 global_freq (B, 1, dim)
        """
        B = x.shape[0]; device = x.device
        tokens, global_freq = self.embed(x)  # (B,20,dim), (B,1,dim)

        if mask_matrix is not None:
            # x-encoder: replace masked tokens with [MASK], keep all 20 positions
            mask_tok = self.mask_token.expand(B, 20, -1)
            # tokens where mask_matrix=False (visible), mask_tok where True
            tokens = torch.where(mask_matrix.unsqueeze(-1), mask_tok, tokens)
            # Full self-attention over global_freq + all 20 tokens
            input_tokens = torch.cat([global_freq, tokens], dim=1)
        else:
            # y-encoder: full original
            input_tokens = torch.cat([global_freq, tokens], dim=1)

        full_out, intermediates, cls_out = self.transformer(input_tokens)

        if mask_matrix is not None:
            # Remove global_freq position, keep all 20
            patch_out = full_out[:, 1:, :]
            trimmed = {}
            for k, v in intermediates.items():
                trimmed[k] = v[:, 1:, :]
            intermediates = trimmed
        else:
            patch_out = full_out[:, 1:, :]
            trimmed = {}
            for k, v in intermediates.items():
                trimmed[k] = v[:, 1:, :]
            intermediates = trimmed

        return patch_out, intermediates, cls_out, global_freq
