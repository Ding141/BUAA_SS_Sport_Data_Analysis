"""Compact deep model for six-class UCI HAR sensor fusion."""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x) + self.shortcut(x)
        return self.pool(x)


class SensorAttention(nn.Module):
    """Learn a light channel gate over accelerometer and gyroscope streams."""
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(4, channels // 2)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=-1)
        weights = self.gate(pooled).unsqueeze(-1)
        return x * weights


class CNNResidualAttention(nn.Module):
    """A small 1D CNN with residual blocks and sensor-channel attention."""
    def __init__(self, in_channels: int = 9, num_classes: int = 6, dropout: float = 0.25):
        super().__init__()
        self.attention = SensorAttention(in_channels)
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64, 7, dropout),
            ConvBlock(64, 128, 5, dropout),
            ConvBlock(128, 192, 3, dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.encoder(x)
        return self.head(x)


def build_model(in_channels: int = 9, num_classes: int = 6, dropout: float = 0.25) -> nn.Module:
    return CNNResidualAttention(in_channels=in_channels, num_classes=num_classes, dropout=dropout)

