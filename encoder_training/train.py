"""Main training loop for masked prediction encoder pretraining.

Architecture:
  x-encoder (student): sees masked input → visible patch embeddings
  y-encoder (teacher): sees full original → target embeddings at masked positions
  Predictor: x-encoder context → predicts y-encoder targets
  SigReg: prevents collapse via sigmoid variance+entropy regularization

y-encoder is EMA-updated from x-encoder with stop-gradient.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import Config
from data.dataset import WISDMWindowDataset
from data.augment import apply_mask
from models.encoder import MaskedEncoder
from models.predictor import Predictor, SigRegHead


class EMAModel:
    """Exponential Moving Average wrapper for y-encoder."""

    def __init__(self, model: nn.Module, tau: float = 0.999):
        self.model = model
        self.tau = tau
        # Initialize EMA weights with model weights
        self.shadow = {name: param.data.clone().detach()
                       for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self, source_model: nn.Module):
        """Update EMA weights: shadow = tau * shadow + (1-tau) * source."""
        for name, param in source_model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = self.tau * self.shadow[name] + \
                                    (1 - self.tau) * param.data.clone().detach()

    def apply(self):
        """Copy shadow weights to model."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


def collate_fn(batch):
    """Custom collate: stack tensors from dict."""
    x = torch.stack([item['x'] for item in batch])  # (B, 6, 200)
    labels = torch.tensor([item['label_idx'] for item in batch])
    return x, labels


def get_cosine_schedule(step, total_steps, start_val, end_val):
    """Cosine schedule from start_val to end_val."""
    progress = step / max(total_steps - 1, 1)
    return end_val + 0.5 * (start_val - end_val) * (1 + math.cos(math.pi * progress))


def train(config: Config):
    device = torch.device(config.train.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Data ----
    train_dataset = WISDMWindowDataset(
        raw_dir=os.path.join(config.data.dataset_root, config.data.raw_dir),
        subject_ids=config.data.train_subjects,
        window_size=config.data.window_size,
        window_stride=config.data.window_stride,
        sensors=config.data.sensors,
    )
    val_dataset = WISDMWindowDataset(
        raw_dir=os.path.join(config.data.dataset_root, config.data.raw_dir),
        subject_ids=config.data.val_subjects,
        window_size=config.data.window_size,
        window_stride=config.data.window_stride,
        sensors=config.data.sensors,
    )
    print(f"Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.train.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, drop_last=False,
    )

    # ---- Models ----
    cfg_enc = config.encoder
    x_encoder = MaskedEncoder(cfg_enc).to(device)
    y_encoder = MaskedEncoder(cfg_enc).to(device)
    predictor = Predictor(
        embed_dim=cfg_enc.embed_dim,
        n_layers=cfg_enc.predictor_n_layers,
        n_heads=cfg_enc.n_heads,
        mlp_hidden=cfg_enc.predictor_mlp_hidden,
    ).to(device)
    sigreg_head = SigRegHead(embed_dim=cfg_enc.embed_dim).to(device)

    # Initialize y_encoder with x_encoder weights
    y_encoder.load_state_dict(x_encoder.state_dict())
    ema = EMAModel(y_encoder, tau=config.train.ema_tau_start)

    # ---- Optimizer ----
    params = (list(x_encoder.parameters()) + list(predictor.parameters())
              + list(sigreg_head.parameters()))
    optimizer = torch.optim.AdamW(params, lr=config.train.lr,
                                  weight_decay=config.train.weight_decay)

    total_steps = config.train.epochs * len(train_loader)
    warmup_steps = config.train.warmup_epochs * len(train_loader)

    # ---- Training loop ----
    global_step = 0
    best_val_loss = float('inf')

    os.makedirs(config.train.save_dir, exist_ok=True)

    for epoch in range(config.train.epochs):
        x_encoder.train()
        predictor.train()
        sigreg_head.train()
        y_encoder.eval()  # teacher always eval

        epoch_pred_loss = 0.0
        epoch_sigreg_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.train.epochs}")
        for batch_idx, (x, labels) in enumerate(pbar):
            x = x.to(device)  # (B, 6, 200)

            # ---- Forward: masking ----
            x_masked, x_original, mask_matrix = apply_mask(x, config.mask)

            # ---- Forward: y-encoder (teacher, no grad) ----
            with torch.no_grad():
                # y-encoder sees full original signal
                y_patch_out, y_intermediates, y_cls, y_global = y_encoder(x_original, mask_matrix=None)

                # Extract targets at masked positions for each target layer
                y_targets = {}
                for layer_idx in cfg_enc.target_layers:
                    target = y_intermediates[layer_idx]  # (B, 20, dim) — all positions
                    # Pick masked positions
                    masked_targets = []
                    for b in range(target.size(0)):
                        mt = target[b][mask_matrix[b]]
                        masked_targets.append(mt)
                    # Pad
                    max_m = max(mt.size(0) for mt in masked_targets)
                    if max_m == 0:
                        continue
                    padded_t = torch.zeros(x.size(0), max_m, target.size(-1), device=device)
                    for b, mt in enumerate(masked_targets):
                        if mt.size(0) > 0:
                            padded_t[b, :mt.size(0)] = mt
                    # LayerNorm target (prevent predictor from cheating via scale)
                    y_targets[layer_idx] = F.layer_norm(padded_t, (padded_t.size(-1),))

            # ---- Forward: x-encoder (student, with mask) ----
            x_patch_out, x_intermediates, x_cls, x_global = x_encoder(x_masked, mask_matrix)

            # ---- Forward: Predictor ----
            preds = predictor(x_patch_out, mask_matrix)  # (B, N_masked, dim)

            # ---- Loss: prediction (L_pred) ----
            L_pred = torch.tensor(0.0, device=device)
            for layer_idx, weight in zip(cfg_enc.target_layers, cfg_enc.target_layer_weights):
                if layer_idx in y_targets:
                    yt = y_targets[layer_idx]  # (B, N_masked, dim)
                    # Align sequence lengths
                    min_m = min(preds.size(1), yt.size(1))
                    delta = F.mse_loss(preds[:, :min_m, :], yt[:, :min_m, :])
                    L_pred = L_pred + weight * delta

            # ---- Loss: SigReg ----
            z_sigmoid = sigreg_head(preds)
            L_sigreg = sigreg_head.compute_loss(
                z_sigmoid, config.train.lambda_var, config.train.lambda_ent
            )

            L_total = L_pred + L_sigreg

            # ---- Backward ----
            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(x_encoder.parameters()) + list(predictor.parameters()),
                config.train.grad_clip
            )
            optimizer.step()

            # ---- EMA update y-encoder ----
            with torch.no_grad():
                ema.update(x_encoder)

            # ---- LR schedule ----
            if global_step < warmup_steps:
                lr = config.train.lr * (global_step + 1) / warmup_steps
            else:
                lr = get_cosine_schedule(
                    global_step - warmup_steps,
                    total_steps - warmup_steps,
                    config.train.lr, config.train.min_lr
                )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # ---- EMA tau schedule ----
            ema.tau = get_cosine_schedule(
                global_step, total_steps,
                config.train.ema_tau_start, config.train.ema_tau_end
            )

            epoch_pred_loss += L_pred.item()
            epoch_sigreg_loss += L_sigreg.item()

            # Logging
            if global_step % config.train.log_interval == 0:
                # Compute std stats for collapse monitoring
                with torch.no_grad():
                    pred_std = preds.std(dim=[0, 1]).mean().item()
                    y_std = y_patch_out.std(dim=[0, 1]).mean().item()
                    x_std = x_patch_out.std(dim=[0, 1]).mean().item()

                pbar.set_postfix({
                    'L_pred': f'{L_pred.item():.4f}',
                    'L_sig': f'{L_sigreg.item():.4f}',
                    'lr': f'{lr:.2e}',
                    'τ': f'{ema.tau:.4f}',
                    'σ_pred': f'{pred_std:.3f}',
                })

            global_step += 1

        # ---- Epoch summary ----
        avg_pred = epoch_pred_loss / len(train_loader)
        avg_sigreg = epoch_sigreg_loss / len(train_loader)
        print(f"Epoch {epoch+1}: L_pred={avg_pred:.4f}, L_sigreg={avg_sigreg:.4f}")

        # ---- Validation ----
        val_loss = validate(val_loader, x_encoder, y_encoder, predictor,
                            sigreg_head, ema, config, device)
        print(f"Val loss: {val_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'x_encoder': x_encoder.state_dict(),
            'y_encoder': y_encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'sigreg_head': sigreg_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
        }
        torch.save(checkpoint, os.path.join(config.train.save_dir, 'last.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(config.train.save_dir, 'best.pt'))
            print(f"  New best model saved!")

        # Apply EMA weights to y_encoder (for next epoch's targets)
        ema.apply()


@torch.no_grad()
def validate(val_loader, x_encoder, y_encoder, predictor, sigreg_head, ema, config, device):
    x_encoder.eval()
    predictor.eval()
    y_encoder.eval()

    total_loss = 0.0
    ema.apply()  # Use EMA weights for validation

    for x, labels in val_loader:
        x = x.to(device)
        x_masked, x_original, mask_matrix = apply_mask(x, config.mask)

        y_patch_out, y_intermediates, y_cls, y_global = y_encoder(x_original, mask_matrix=None)

        y_targets = {}
        for layer_idx in config.encoder.target_layers:
            if layer_idx not in y_intermediates:
                continue
            target = y_intermediates[layer_idx]
            masked_targets = []
            for b in range(target.size(0)):
                mt = target[b][mask_matrix[b]]
                masked_targets.append(mt)
            max_m = max(mt.size(0) for mt in masked_targets)
            if max_m == 0:
                continue
            padded_t = torch.zeros(x.size(0), max_m, target.size(-1), device=device)
            for b, mt in enumerate(masked_targets):
                if mt.size(0) > 0:
                    padded_t[b, :mt.size(0)] = mt
            y_targets[layer_idx] = F.layer_norm(padded_t, (padded_t.size(-1),))

        x_patch_out, _, _, _ = x_encoder(x_masked, mask_matrix)
        preds = predictor(x_patch_out, mask_matrix)

        L_pred = torch.tensor(0.0, device=device)
        for layer_idx, weight in zip(config.encoder.target_layers,
                                      config.encoder.target_layer_weights):
            if layer_idx in y_targets:
                yt = y_targets[layer_idx]
                min_m = min(preds.size(1), yt.size(1))
                L_pred = L_pred + weight * F.mse_loss(preds[:, :min_m, :], yt[:, :min_m, :])

        z = sigreg_head(preds)
        L_sigreg = sigreg_head.compute_loss(z, config.train.lambda_var, config.train.lambda_ent)
        total_loss += (L_pred + L_sigreg).item()

    return total_loss / len(val_loader)


if __name__ == '__main__':
    config = Config()
    train(config)
