"""Full pipeline v2: multi-mask, x-encoder mask tokens, LMM freq loss, SupCon contrastive.
Pretrain → save model → linear probe → report results. Target: within 20 min."""

import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn, torch.nn.functional as F, time, copy
from torch.utils.data import DataLoader
from config import Config
from data.dataset import WISDMWindowDataset
from data.augment import multi_mask
from models.encoder import MaskedEncoder
from models.predictor import Predictor, LightReconHead, lmm_loss, SigRegHead
from train import EMAModel, get_cosine_schedule

def log(*args):
    print(*args, flush=True)

# ============================================================
# Config
# ============================================================
cfg = Config()
cfg.data.train_subjects = [1600,1601,1602,1603,1604,1605]
cfg.data.val_subjects   = [1606,1607]
cfg.data.test_subjects  = [1608,1609]
cfg.data.window_stride = 50
cfg.train.epochs = 12        # fewer epochs, but M=4 multi-mask per step
cfg.train.batch_size = 128   # smaller batch, M=4 views
cfg.train.lr = 3e-4
cfg.train.min_lr = 1e-5
cfg.train.warmup_epochs = 2
cfg.train.lambda_var = 0.15
cfg.train.lambda_ent = 0.01
cfg.train.device = 'cuda'
cfg.encoder.embed_dim = 192
cfg.encoder.n_layers = 4
cfg.encoder.n_heads = 6
cfg.encoder.mlp_ratio = 3.0
cfg.encoder.target_layers = [2, 4]
cfg.encoder.target_layer_weights = [0.3, 1.0]
cfg.encoder.predictor_n_layers = 1
cfg.encoder.predictor_mlp_hidden = 384

device = 'cuda'; ce = cfg.encoder; tc = cfg.train
MULTI_MASK = 4
LAMBDA_LMM = 0.1
LAMBDA_SUPCON = 0.15
SUPCON_TEMP = 0.07

# ============================================================
# Data
# ============================================================
def collate_sup(b):
    x = torch.stack([it['x'] for it in b])
    y = torch.tensor([it['label_idx'] for it in b], dtype=torch.long)
    return x, y

log('Loading data...')
t0 = time.time()
base = os.path.join(cfg.data.dataset_root, cfg.data.raw_dir)
train_ds = WISDMWindowDataset(base, cfg.data.train_subjects,
    cfg.data.window_size, cfg.data.window_stride, cfg.data.sensors)
val_ds = WISDMWindowDataset(base, cfg.data.val_subjects,
    cfg.data.window_size, cfg.data.window_stride, cfg.data.sensors)
test_ds = WISDMWindowDataset(base, cfg.data.test_subjects,
    cfg.data.window_size, cfg.data.window_stride, cfg.data.sensors)
log(f'Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}  ({time.time()-t0:.1f}s)')

tl = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True, collate_fn=collate_sup, drop_last=True)
vl = DataLoader(val_ds, batch_size=tc.batch_size, shuffle=False, collate_fn=collate_sup, drop_last=False)
test_l = DataLoader(test_ds, batch_size=tc.batch_size, shuffle=False, collate_fn=collate_sup, drop_last=False)

# ============================================================
# Models
# ============================================================
x_enc = MaskedEncoder(ce).to(device)
y_enc = MaskedEncoder(ce).to(device)
pred = Predictor(ce.embed_dim, ce.predictor_n_layers, ce.n_heads, ce.predictor_mlp_hidden).to(device)
recon_head = LightReconHead(ce.embed_dim, ce.n_channels * ce.patch_size, 128).to(device)
sigreg = SigRegHead(ce.embed_dim).to(device)
y_enc.load_state_dict(x_enc.state_dict())
ema = EMAModel(y_enc, tau=tc.ema_tau_start)

# SupCon projection head (MLP on CLS token)
supcon_proj = nn.Sequential(
    nn.Linear(ce.embed_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
).to(device)

params = (list(x_enc.parameters()) + list(pred.parameters())
          + list(recon_head.parameters()) + list(sigreg.parameters())
          + list(supcon_proj.parameters()))
n_p = sum(p.numel() for p in params)
log(f'Total params: {n_p/1e6:.2f}M')

opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
total_steps = tc.epochs * len(tl)
warmup_steps = tc.warmup_epochs * len(tl)
log(f'Steps/epoch: {len(tl)}  Total: {total_steps}  Warmup: {warmup_steps}')

# ============================================================
# Phase 1: Pretraining
# ============================================================
log(f'\n{"="*60}')
log(f'Phase 1: Pretraining (multi-mask={MULTI_MASK}, LMM, SupCon)')
log(f'{"="*60}')

gs = 0; start_time = time.time()
for epoch in range(tc.epochs):
    x_enc.train(); pred.train(); recon_head.train(); sigreg.train()
    supcon_proj.train(); y_enc.eval()
    ep_loss = 0.0; ep_l_pred = 0.0; ep_l_lmm = 0.0; ep_l_supcon = 0.0

    for x, labels in tl:
        B = x.size(0)
        x_orig_gpu = x.to(device)
        labels_gpu = labels.to(device)

        # ---- Generate M masked views + 1 original ----
        views, x_orig_copy = multi_mask(x, cfg.mask, M=MULTI_MASK)

        # ---- Teacher (y-encoder): 1 forward pass on full signal ----
        with torch.no_grad():
            _, y_ints, _, _ = y_enc(x_orig_gpu, None)
            y_targets = {}
            for lidx in ce.target_layers:
                if lidx not in y_ints: continue
                y_targets[lidx] = F.layer_norm(y_ints[lidx].float(), (y_ints[lidx].size(-1),))

        # ---- Student: M forward passes (shared teacher target) ----
        L_pred_total = torch.tensor(0.0, device=device)
        L_lmm_total = torch.tensor(0.0, device=device)

        for m in range(MULTI_MASK):
            x_masked, mm = views[m]
            x_masked = x_masked.to(device)
            mm = mm.to(device)

            # x-encoder with mask tokens
            x_patch_out, x_ints, _, _ = x_enc(x_masked, mm)

            # Predictor
            preds = pred(x_patch_out, mm)  # (B, N_m, dim)

            # L_pred: match teacher targets at masked positions
            L_pred_m = torch.tensor(0.0, device=device)
            for lidx, w in zip(ce.target_layers, ce.target_layer_weights):
                if lidx not in y_ints: continue
                yt = y_targets[lidx]  # (B, 20, dim)
                # Extract masked positions from teacher
                mt = [yt[b][mm[b]] for b in range(B)]
                max_mm = max(mt[b].size(0) for b in range(B))
                if max_mm == 0: continue
                yt_pad = torch.zeros(B, max_mm, yt.size(-1), device=device)
                for b, m in enumerate(mt):
                    if m.size(0) > 0: yt_pad[b,:m.size(0)] = m
                n_pred = min(preds.size(1), yt_pad.size(1))
                L_pred_m = L_pred_m + w * F.mse_loss(preds[:,:n_pred], yt_pad[:,:n_pred])
            L_pred_total = L_pred_total + L_pred_m / MULTI_MASK

            # L_lmm: frequency-aware loss
            with torch.no_grad():
                # Original signal for masked patches
                x_patches = x_orig_gpu.view(B, ce.n_channels, 20, ce.patch_size)
                x_patches = x_patches.permute(0, 2, 1, 3).reshape(B, 20, -1)  # (B,20,60)
                target_patches = []
                for b in range(B):
                    target_patches.append(x_patches[b][mm[b]])
                max_tp = max(tp.size(0) for tp in target_patches)
                if max_tp > 0:
                    tp_pad = torch.zeros(B, max_tp, x_patches.size(-1), device=device)
                    for b, tp in enumerate(target_patches):
                        if tp.size(0) > 0: tp_pad[b,:tp.size(0)] = tp
                    recon = recon_head(preds)  # (B, N_m, 60)
                    nr = min(recon.size(1), tp_pad.size(1))
                    L_lmm_total = L_lmm_total + lmm_loss(recon[:,:nr], tp_pad[:,:nr]) / MULTI_MASK

        # ---- SupCon loss on CLS tokens (full signal, no mask) ----
        _, _, cls_full, _ = x_enc(x_orig_gpu, None)  # full signal for class representation
        cls_proj = F.normalize(supcon_proj(cls_full), dim=-1)  # (B, 64)

        # SupCon: positives = same label
        sim = torch.matmul(cls_proj, cls_proj.T) / SUPCON_TEMP  # (B, B)
        label_eq = (labels_gpu.unsqueeze(0) == labels_gpu.unsqueeze(1)).float()  # (B, B)
        # Remove diagonal
        mask_pos = label_eq - torch.eye(B, device=device)

        exp_sim = torch.exp(sim)
        # For each anchor: sum over positives / sum over all (except self)
        pos_sum = (exp_sim * mask_pos).sum(dim=1)
        all_sum = (exp_sim * (1 - torch.eye(B, device=device))).sum(dim=1)
        # Only compute where there are positives
        n_pos = mask_pos.sum(dim=1)
        valid = n_pos > 0
        L_supcon = torch.tensor(0.0, device=device)
        if valid.any():
            L_supcon = -(torch.log(pos_sum[valid] / all_sum[valid].clamp(min=1e-8))).mean()

        # ---- SigReg ----
        z = sigreg(preds)
        L_sig = sigreg.compute_loss(z, tc.lambda_var, tc.lambda_ent)

        L_total = L_pred_total + L_sig + LAMBDA_LMM * L_lmm_total + LAMBDA_SUPCON * L_supcon

        opt.zero_grad(); L_total.backward()
        torch.nn.utils.clip_grad_norm_(list(x_enc.parameters())+list(pred.parameters()), 1.0)
        opt.step()

        with torch.no_grad(): ema.update(x_enc)

        # LR schedule
        if gs < warmup_steps:
            lr = tc.lr * (gs+1) / warmup_steps
        else:
            lr = get_cosine_schedule(gs-warmup_steps, total_steps-warmup_steps, tc.lr, tc.min_lr)
        for pg in opt.param_groups: pg['lr'] = lr
        ema.tau = get_cosine_schedule(gs, total_steps, tc.ema_tau_start, tc.ema_tau_end)

        ep_loss += L_pred_total.item()
        ep_l_lmm += L_lmm_total.item()
        ep_l_supcon += L_supcon.item()
        gs += 1

    avg = ep_loss/len(tl); al = ep_l_lmm/len(tl); ac = ep_l_supcon/len(tl)
    dt = time.time() - start_time
    log(f'  Epoch {epoch+1:2d} | L_pred={avg:.4f}  L_lmm={al:.4f}  L_sup={ac:.4f}  [{dt:.0f}s]')

# Save checkpoint
ckpt_path = os.path.join(cfg.train.save_dir, 'pretrained_v2.pt')
os.makedirs(cfg.train.save_dir, exist_ok=True)
torch.save({'x_encoder': x_enc.state_dict()}, ckpt_path)
log(f'\nPretraining done in {time.time()-start_time:.0f}s, saved to {ckpt_path}')

# ============================================================
# Phase 2: Linear Probing
# ============================================================
log(f'\n{"="*60}')
log(f'Phase 2: Linear Probe')
log(f'{"="*60}')

# Freeze x-encoder
for p in x_enc.parameters():
    p.requires_grad = False
x_enc.eval()

n_classes = 18
classifier = nn.Sequential(
    nn.Linear(ce.embed_dim, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, n_classes),
).to(device)

log(f'Classifier params: {sum(p.numel() for p in classifier.parameters()):,}')

opt_cls = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cls, T_max=30)

best_val_acc = 0.0; best_state = None

for epoch in range(30):
    classifier.train()
    tr_loss = 0.0; tr_correct = 0; tr_total = 0
    for x, y in tl:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, _, cls_out, _ = x_enc(x, None)
        logits = classifier(cls_out)
        loss = F.cross_entropy(logits, y)
        opt_cls.zero_grad(); loss.backward(); opt_cls.step()
        tr_loss += loss.item(); tr_correct += (logits.argmax(1)==y).sum().item(); tr_total += y.size(0)
    scheduler.step()

    classifier.eval()
    val_correct = 0; val_total = 0
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(device), y.to(device)
            _, _, cls_out, _ = x_enc(x, None)
            logits = classifier(cls_out)
            val_correct += (logits.argmax(1)==y).sum().item(); val_total += y.size(0)
    val_acc = val_correct/val_total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(classifier.state_dict())
    if epoch % 3 == 0:
        log(f'  Probe {epoch:2d} | train_acc={tr_correct/tr_total:.3f}  val_acc={val_acc:.3f}')

# ---- Test ----
classifier.load_state_dict(best_state)
classifier.eval()
test_correct = 0; test_total = 0
all_preds = []; all_labels = []
with torch.no_grad():
    for x, y in test_l:
        x, y = x.to(device), y.to(device)
        _, _, cls_out, _ = x_enc(x, None)
        logits = classifier(cls_out)
        preds = logits.argmax(1)
        test_correct += (preds==y).sum().item(); test_total += y.size(0)
        all_preds.append(preds.cpu()); all_labels.append(y.cpu())

test_acc = test_correct/test_total
all_preds = torch.cat(all_preds); all_labels = torch.cat(all_labels)

from collections import Counter
pcc = Counter(); pct = Counter()
for p, l in zip(all_preds.tolist(), all_labels.tolist()):
    pct[l] += 1
    if p == l: pcc[l] += 1

act_names = {0:'A-walk',1:'B-jog',2:'C-stairs',3:'D-sit',4:'E-stand',5:'F-type',
    6:'G-teeth',7:'H-soup',8:'I-chips',9:'J-pasta',10:'K-drink',11:'L-sandw',
    12:'M-kick',13:'O-catch',14:'P-dribble',15:'Q-write',16:'R-clap',17:'S-fold'}

log(f'\n{"="*60}')
log(f'RESULTS')
log(f'{"="*60}')
log(f'Random baseline: {1/n_classes:.1%}')
log(f'Test Accuracy:  {test_acc:.2%} ({test_correct}/{test_total})')
log(f'Best Val Acc:   {best_val_acc:.2%}')
log(f'\nPer-class:')
for cls_idx in sorted(pct.keys()):
    acc = pcc[cls_idx]/pct[cls_idx]
    name = act_names.get(cls_idx, '?')
    bar = '#'*int(acc*40) + ' ' + '-'*max(0,40-int(acc*40))
    log(f'  {name:>12s}: {acc:.2%} |{bar}|')
log(f'\nTotal: {time.time()-start_time:.0f}s')
