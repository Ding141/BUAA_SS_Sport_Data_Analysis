"""Predictor + LightReconHead for frequency-aware loss + SigReg head."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """Predicts masked-position representations from x-encoder full-sequence output."""

    def __init__(self, embed_dim=256, n_layers=2, n_heads=8, mlp_hidden=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed_full = nn.Parameter(torch.randn(1, 20, embed_dim) * 0.02)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                dim_feedforward=mlp_hidden, dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_out: torch.Tensor, mask_matrix: torch.Tensor):
        """
        Args:
            x_out: (B, 20, dim) x-encoder full-sequence output (visible + mask tokens)
            mask_matrix: (B, 20) bool, True=masked
        Returns:
            preds: (B, N_masked_max, dim) predictions at masked positions
        """
        B, _, dim = x_out.shape
        device = x_out.device
        x = x_out + self.pos_embed_full
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.final_norm(x)

        preds = []
        for b in range(B):
            preds.append(x[b][mask_matrix[b]])
        max_m = max(p.size(0) for p in preds)
        if max_m == 0:
            max_m = 1
        padded = torch.zeros(B, max_m, dim, device=device)
        for b, p in enumerate(preds):
            if p.size(0) > 0:
                padded[b, :p.size(0)] = p
        return padded


class LightReconHead(nn.Module):
    """Lightweight MLP: predicted embedding → raw patch signal for LMM loss.

    Input: (B, N_masked, embed_dim)
    Output: (B, N_masked, patch_flat_dim)  where patch_flat_dim = n_channels * patch_size
    """

    def __init__(self, embed_dim=256, patch_flat_dim=60, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, patch_flat_dim),
        )

    def forward(self, x):
        return self.net(x)


def lmm_loss(recon: torch.Tensor, target: torch.Tensor, eps: float = 1e-4):
    """Log-Magnitude Mean loss: compare log-scale FFT magnitude spectra.

    Args:
        recon: (B, M, patch_flat_dim)
        target: (B, M, patch_flat_dim)
    """
    # Reshape to per-patch signals for FFT
    B, M, D = recon.shape
    # Compute FFT magnitude along last dim
    spec_recon = torch.abs(torch.fft.rfft(recon, dim=-1))  # (B, M, F)
    spec_target = torch.abs(torch.fft.rfft(target, dim=-1))

    # Mean over the spatial dim (equivalent to average magnitude per freq bin)
    mag_recon = spec_recon.mean(dim=1)   # (B, F)
    mag_target = spec_target.mean(dim=1) # (B, F)

    L = F.mse_loss(torch.log(mag_recon + eps), torch.log(mag_target + eps))
    return L


class SigRegHead(nn.Module):
    """Sigmoid-based regularization head."""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        return torch.sigmoid(self.alpha * x_norm)

    def compute_loss(self, z, lambda_var=0.1, lambda_ent=0.01):
        z_flat = z.view(-1, z.size(-1))
        std_per_dim = z_flat.std(dim=0)
        L_var = F.relu(0.1 - std_per_dim).mean()
        eps = 1e-8
        entropy = -(z * torch.log(z + eps) + (1 - z) * torch.log(1 - z + eps))
        ent_per_dim = entropy.mean(dim=[0, 1])
        L_ent = F.relu(0.3 - ent_per_dim).mean()
        return lambda_var * L_var + lambda_ent * L_ent
