"""FeatureFusionNet: 双分支多传感器融合网络。

加速度计和陀螺仪特征分别经独立MLP编码，再通过四路融合
concat(a, g, |a-g|, a*g) 实现显式跨传感器信息交互。
"""
import torch
from torch import nn


class SensorFeatureBranch(nn.Module):
    """单传感器特征编码分支: n_features → hidden → out_dim."""

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
    """双分支特征融合网络。

    输入: accel 78维 + gyro 78维 → 各自编码至80维 →
          concat(a,g,|a-g|,a*g) → 320维 → MLP分类头。
    """

    def __init__(self, accel_dim: int, gyro_dim: int, n_classes: int = 6) -> None:
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
