from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: bool = True) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        ]
        if pool:
            layers.append(nn.MaxPool1d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1D(nn.Module):
    def __init__(self, n_channels: int = 6, n_classes: int = 18) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(n_channels, 64, 7),
            ConvBlock(64, 128, 5),
            ConvBlock(128, 192, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


class CNNGRU(nn.Module):
    def __init__(self, n_channels: int = 6, n_classes: int = 18, bidirectional: bool = False) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(n_channels, 64, 7),
            ConvBlock(64, 128, 5),
            ConvBlock(128, 128, 3, pool=False),
        )
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = 128 if bidirectional else 64
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x).transpose(1, 2)
        sequence, _ = self.gru(x)
        pooled = sequence.mean(dim=1)
        return self.classifier(pooled)


class CNNBiGRU(CNNGRU):
    def __init__(self, n_channels: int = 6, n_classes: int = 18) -> None:
        super().__init__(n_channels=n_channels, n_classes=n_classes, bidirectional=True)


class CNNLSTM(nn.Module):
    def __init__(self, n_channels: int = 6, n_classes: int = 18, bidirectional: bool = False) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(n_channels, 64, 7),
            ConvBlock(64, 128, 5),
            ConvBlock(128, 128, 3, pool=False),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = 128 if bidirectional else 64
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x).transpose(1, 2)
        sequence, _ = self.lstm(x)
        return self.classifier(sequence.mean(dim=1))


class CNNBiLSTM(CNNLSTM):
    def __init__(self, n_channels: int = 6, n_classes: int = 18) -> None:
        super().__init__(n_channels=n_channels, n_classes=n_classes, bidirectional=True)


class SensorTransformer(nn.Module):
    """Lightweight Transformer encoder for short inertial-sensor windows."""

    def __init__(self, n_channels: int = 6, n_classes: int = 18, d_model: int = 128) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            ConvBlock(n_channels, 64, 7),
            ConvBlock(64, d_model, 5),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + 50, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.20,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.30),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, : x.shape[1], :]
        x = self.transformer(x)
        return self.classifier(x[:, 0])


class DualBranchCNNGRU(nn.Module):
    """Late-fusion model with separate accel and gyro branches before GRU."""

    def __init__(self, n_channels: int = 6, n_classes: int = 18) -> None:
        super().__init__()
        if n_channels != 6:
            raise ValueError("DualBranchCNNGRU expects 6 channels: accel_xyz + gyro_xyz")
        self.accel_encoder = nn.Sequential(ConvBlock(3, 48, 7), ConvBlock(48, 64, 5))
        self.gyro_encoder = nn.Sequential(ConvBlock(3, 48, 7), ConvBlock(48, 64, 5))
        self.fusion = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        accel = self.accel_encoder(x[:, :3, :])
        gyro = self.gyro_encoder(x[:, 3:, :])
        fused = self.fusion(torch.cat([accel, gyro], dim=1)).transpose(1, 2)
        sequence, _ = self.gru(fused)
        return self.classifier(sequence.mean(dim=1))


# ── Feature‑based models ───────────────────────────────────────────


class SensorFeatureBranch(nn.Module):
    """Single-sensor feature encoder: n_features → hidden → out_dim."""

    def __init__(self, n_features: int, hidden: int = 160, out_dim: int = 80) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureFusionNet(nn.Module):
    """Two-branch accel/gyro fusion network for hand-crafted features.

    Each sensor modality is encoded independently, then fused via
    concat(a, g, |a-g|, a*g) → 320-D → MLP classifier.
    """

    def __init__(self, accel_dim: int, gyro_dim: int, n_classes: int) -> None:
        super().__init__()
        self.accel_branch = SensorFeatureBranch(accel_dim)
        self.gyro_branch = SensorFeatureBranch(gyro_dim)
        fused_dim = 80 * 4  # a(80) + g(80) + |a-g|(80) + a*g(80)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, n_classes),
        )

    def forward(self, accel: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        a = self.accel_branch(accel)
        g = self.gyro_branch(gyro)
        fused = torch.cat([a, g, torch.abs(a - g), a * g], dim=1)
        return self.classifier(fused)


class FeatureMLP(nn.Module):
    """Single-branch MLP over concatenated accel+gyro features."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureResNet(nn.Module):
    """Residual MLP over hand-crafted features."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, 128)
        self.block1 = nn.Sequential(
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.20),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.20),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.classifier(h)


def build_deep_model(model_name: str, n_channels: int, n_classes: int) -> nn.Module:
    if model_name == "cnn1d":
        return CNN1D(n_channels=n_channels, n_classes=n_classes)
    if model_name == "cnn_gru":
        return CNNGRU(n_channels=n_channels, n_classes=n_classes, bidirectional=False)
    if model_name == "cnn_bigru":
        return CNNBiGRU(n_channels=n_channels, n_classes=n_classes)
    if model_name == "cnn_lstm":
        return CNNLSTM(n_channels=n_channels, n_classes=n_classes, bidirectional=False)
    if model_name == "cnn_bilstm":
        return CNNBiLSTM(n_channels=n_channels, n_classes=n_classes)
    if model_name == "transformer":
        return SensorTransformer(n_channels=n_channels, n_classes=n_classes)
    if model_name == "dual_branch_bigru":
        return DualBranchCNNGRU(n_channels=n_channels, n_classes=n_classes)
    raise ValueError(
        "Unknown model_name. Use cnn1d, cnn_gru, cnn_bigru, cnn_lstm, "
        "cnn_bilstm, transformer, or dual_branch_bigru"
    )
