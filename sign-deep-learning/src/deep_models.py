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
    def __init__(self, n_channels: int = 6, n_classes: int = 12) -> None:
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
    def __init__(self, n_channels: int = 6, n_classes: int = 12, bidirectional: bool = False) -> None:
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
    def __init__(self, n_channels: int = 6, n_classes: int = 12) -> None:
        super().__init__(n_channels=n_channels, n_classes=n_classes, bidirectional=True)


class CNNLSTM(nn.Module):
    def __init__(self, n_channels: int = 6, n_classes: int = 12, bidirectional: bool = False) -> None:
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
    def __init__(self, n_channels: int = 6, n_classes: int = 12) -> None:
        super().__init__(n_channels=n_channels, n_classes=n_classes, bidirectional=True)


class SensorTransformer(nn.Module):
    """Lightweight Transformer encoder for short inertial-sensor windows."""

    def __init__(self, n_channels: int = 6, n_classes: int = 12, d_model: int = 128) -> None:
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

    def __init__(self, n_channels: int = 6, n_classes: int = 12) -> None:
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


# ---------------------------------------------------------------------------
# New architectures for broader model comparison
# ---------------------------------------------------------------------------

class TCNBlock(nn.Module):
    """Dilated convolution block with residual connection for TCN.

    Applies padding proportional to dilation, then crops the output back
    to the original sequence length so the residual can be added directly.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad_amount = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad_amount, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=self.pad_amount, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        downsample_layers: list[nn.Module] = []
        if in_ch != out_ch:
            downsample_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
        self.downsample = nn.Sequential(*downsample_layers) if downsample_layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        L_in = x.shape[-1]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        # Crop dilated padding excess to keep length = L_in
        out = out[..., :L_in]
        return self.relu(out + residual[..., :L_in])


class TCN(nn.Module):
    """Temporal Convolutional Network with dilated residual blocks.

    Designed to capture multi-scale temporal dependencies through
    exponentially increasing dilation rates, without recurrence.
    """

    def __init__(self, n_channels: int = 6, n_classes: int = 12,
                 hidden_dim: int = 64, num_levels: int = 5,
                 kernel_size: int = 7, dropout: float = 0.20) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_channels
        for level in range(num_levels):
            dilation = 2 ** level
            out_dim = hidden_dim * min(2, level + 1) if level < 2 else hidden_dim * 2
            layers.append(TCNBlock(in_dim, out_dim, kernel_size, dilation, dropout))
            in_dim = out_dim
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(in_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.tcn(x)))


class InceptionModule(nn.Module):
    """Multi-scale 1D convolution block — the core building block of InceptionTime."""

    def __init__(self, in_ch: int, bottleneck_ch: int, out_per_branch: int) -> None:
        super().__init__()
        # Bottleneck
        self.bottleneck = nn.Conv1d(in_ch, bottleneck_ch, kernel_size=1) if in_ch > 1 else nn.Identity()
        actual_in = bottleneck_ch if in_ch > 1 else in_ch

        # Three parallel conv branches with different receptive fields
        self.branch10 = nn.Sequential(
            nn.Conv1d(actual_in, out_per_branch, kernel_size=9, padding=4), nn.BatchNorm1d(out_per_branch), nn.ReLU(),
        )
        self.branch20 = nn.Sequential(
            nn.Conv1d(actual_in, out_per_branch, kernel_size=19, padding=9), nn.BatchNorm1d(out_per_branch), nn.ReLU(),
        )
        self.branch40 = nn.Sequential(
            nn.Conv1d(actual_in, out_per_branch, kernel_size=39, padding=19), nn.BatchNorm1d(out_per_branch), nn.ReLU(),
        )
        # MaxPool branch
        self.branch_mp = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_per_branch, kernel_size=1),
            nn.BatchNorm1d(out_per_branch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        b10 = self.branch10(z)
        b20 = self.branch20(z)
        b40 = self.branch40(z)
        bmp = self.branch_mp(x)  # maxpool branch bypasses bottleneck
        return torch.cat([b10, b20, b40, bmp], dim=1)


class InceptionTime(nn.Module):
    """InceptionTime: ensemble of multi-scale 1D convolutional modules.

    Reference: Ismail Fawaz et al. "InceptionTime: Finding AlexNet for
    Time Series Classification" (Data Mining and Knowledge Discovery, 2020).
    Adapted here for 6-channel inertial sensor windows.
    """

    def __init__(self, n_channels: int = 6, n_classes: int = 12,
                 n_modules: int = 3, bottleneck_ch: int = 32,
                 out_per_branch: int = 32, dropout: float = 0.20) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_channels)
        modules: list[nn.Module] = []
        in_ch = n_channels
        for i in range(n_modules):
            modules.append(InceptionModule(in_ch, bottleneck_ch, out_per_branch))
            in_ch = out_per_branch * 4  # 4 branches concatenated
        self.inception_stack = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout + 0.10),
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        return self.classifier(self.pool(self.inception_stack(x)))


class DeepConvLSTM(nn.Module):
    """DeepConvLSTM: four Conv1D layers followed by two LSTM layers.

    Classic architecture from Ordóñez & Roggen (Sensors, 2016):
    "Deep Convolutional and LSTM Recurrent Neural Networks for
    Multimodal Wearable Activity Recognition".
    """

    def __init__(self, n_channels: int = 6, n_classes: int = 12,
                 conv_filters: tuple[int, ...] = (64, 64, 128, 128),
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 dropout: float = 0.25) -> None:
        super().__init__()
        # 4 conv blocks
        conv_blocks: list[nn.Module] = []
        in_ch = n_channels
        for i, filters in enumerate(conv_filters):
            pool = i < 2  # pool only first two layers (keep temporal resolution later)
            conv_blocks.append(ConvBlock(in_ch, filters, kernel_size=5, pool=pool))
            in_ch = filters
        self.conv = nn.Sequential(*conv_blocks)
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=conv_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout + 0.10),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout - 0.05),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).transpose(1, 2)  # (N, C, T) -> (N, T, C)
        sequence, _ = self.lstm(x)
        return self.classifier(sequence[:, -1, :])  # last timestep


class ResidualBlock1D(nn.Module):
    """1D residual block with two convolutions and a skip connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 1,
                 dropout: float = 0.20) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return self.dropout(out)


class ResNet1D(nn.Module):
    """Residual 1D CNN adapted for inertial sensor windows.

    Uses bottleneck-style residual blocks with progressive channel
    expansion and temporal downsampling for efficient deep learning.
    """

    def __init__(self, n_channels: int = 6, n_classes: int = 12,
                 base_filters: int = 64, dropout: float = 0.25) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, base_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        # Progressive channel expansion with downsampling
        self.layer1 = ResidualBlock1D(base_filters, base_filters, kernel_size=5, stride=1, dropout=dropout)
        self.layer2 = ResidualBlock1D(base_filters, base_filters * 2, kernel_size=5, stride=2, dropout=dropout)
        self.layer3 = ResidualBlock1D(base_filters * 2, base_filters * 3, kernel_size=5, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout + 0.10),
            nn.Linear(base_filters * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(self.pool(x))


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
    if model_name == "tcn":
        return TCN(n_channels=n_channels, n_classes=n_classes)
    if model_name == "inception_time":
        return InceptionTime(n_channels=n_channels, n_classes=n_classes)
    if model_name == "deep_conv_lstm":
        return DeepConvLSTM(n_channels=n_channels, n_classes=n_classes)
    if model_name == "resnet1d":
        return ResNet1D(n_channels=n_channels, n_classes=n_classes)
    raise ValueError(
        "Unknown model_name. Use cnn1d, cnn_gru, cnn_bigru, cnn_lstm, "
        "cnn_bilstm, transformer, dual_branch_bigru, tcn, inception_time, "
        "deep_conv_lstm, or resnet1d"
    )
