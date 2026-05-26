"""Quick training run: ~8 minutes, validates the architecture."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn.functional as F, time
from torch.utils.data import DataLoader
from config import Config
from data.dataset import WISDMWindowDataset
from data.augment import apply_mask
from models.encoder import MaskedEncoder
from models.predictor import Predictor, SigRegHead
from train import EMAModel, get_cosine_schedule

def collate_fn(b):
    x = torch.stack([it['x'] for it in b])
    return x, torch.tensor([it['label_idx'] for it in b])

# ---- Config ----
cfg = Config()
cfg.data.train_subjects = [1600,1601,1602,1603,1604,1605]
cfg.data.val_subjects   = [1606,1607]
cfg.data.window_stride = 50
cfg.train.epochs = 15
cfg.train.batch_size = 256
cfg.train.lr = 3e-4
cfg.train.min_lr = 1e-5
cfg.train.warmup_epochs = 3
cfg.train.lambda_var = 0.2
cfg.train.lambda_ent = 0.02
cfg.train.device = 'cuda'
cfg.encoder.embed_dim = 192
cfg.encoder.n_layers = 4
cfg.encoder.n_heads = 6
cfg.encoder.mlp_ratio = 3.0
cfg.encoder.target_layers = [2, 4]
cfg.encoder.target_layer_weights = [0.3, 1.0]
cfg.encoder.predictor_n_layers = 1
cfg.encoder.predictor_mlp_hidden = 384

device = 'cuda'
ce = cfg.encoder; tc = cfg.train

def log(*args):
    print(*args, flush=True)

log('Loading data...')
t0 = time.time()
train_ds = WISDMWindowDataset(os.path.join(cfg.data.dataset_root, cfg.data.raw_dir),
    cfg.data.train_subjects, cfg.data.window_size, cfg.data.window_stride, cfg.data.sensors)
val_ds = WISDMWindowDataset(os.path.join(cfg.data.dataset_root, cfg.data.raw_dir),
    cfg.data.val_subjects, cfg.data.window_size, cfg.data.window_stride, cfg.data.sensors)
log(f'Train: {len(train_ds)} windows, Val: {len(val_ds)} windows ({time.time()-t0:.1f}s)')

tl = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
vl = DataLoader(val_ds, batch_size=tc.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

x_enc = MaskedEncoder(ce).to(device)
y_enc = MaskedEncoder(ce).to(device)
pred = Predictor(ce.embed_dim, ce.predictor_n_layers, ce.n_heads, ce.predictor_mlp_hidden).to(device)
sigreg = SigRegHead(ce.embed_dim).to(device)
y_enc.load_state_dict(x_enc.state_dict())
ema = EMAModel(y_enc, tau=tc.ema_tau_start)

params = list(x_enc.parameters()) + list(pred.parameters()) + list(sigreg.parameters())
log(f'Params: {sum(p.numel() for p in params)/1e6:.1f}M')
opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)

total_steps = tc.epochs * len(tl)
warmup_steps = tc.warmup_epochs * len(tl)
log(f'Steps/epoch: {len(tl)}, Total: {total_steps}, Warmup: {warmup_steps}')

gs = 0; start_time = time.time(); history = []

for epoch in range(tc.epochs):
    x_enc.train(); pred.train(); y_enc.eval()
    ep_loss = 0.0; t_ep = time.time()

    for x, labels in tl:
        x = x.to(device)
        x_m, x_orig, mm = apply_mask(x, cfg.mask)

        with torch.no_grad():
            _, y_ints, _, _ = y_enc(x_orig, None)
            yt = {}
            for lidx in ce.target_layers:
                if lidx not in y_ints: continue
                tgt = y_ints[lidx]
                mt = [tgt[b][mm[b]] for b in range(tgt.size(0))]
                max_m = max(mt[b].size(0) for b in range(len(mt)))
                if max_m == 0: continue
                pad = torch.zeros(x.size(0), max_m, tgt.size(-1), device=device)
                for b, m in enumerate(mt):
                    if m.size(0) > 0: pad[b,:m.size(0)] = m
                yt[lidx] = F.layer_norm(pad, (pad.size(-1),))

        x_out, _, _, _ = x_enc(x_m, mm)
        preds = pred(x_out, mm)

        L_pred = torch.tensor(0.0, device=device)
        for lidx, w in zip(ce.target_layers, ce.target_layer_weights):
            if lidx in yt:
                yt_l = yt[lidx]; min_m = min(preds.size(1), yt_l.size(1))
                L_pred = L_pred + w * F.mse_loss(preds[:,:min_m], yt_l[:,:min_m])

        z = sigreg(preds); L_sig = sigreg.compute_loss(z, tc.lambda_var, tc.lambda_ent)
        L_total = L_pred + L_sig; opt.zero_grad(); L_total.backward()
        torch.nn.utils.clip_grad_norm_(list(x_enc.parameters())+list(pred.parameters()), 1.0)
        opt.step()
        with torch.no_grad(): ema.update(x_enc)

        if gs < warmup_steps: lr = tc.lr*(gs+1)/warmup_steps
        else: lr = get_cosine_schedule(gs-warmup_steps, total_steps-warmup_steps, tc.lr, tc.min_lr)
        for pg in opt.param_groups: pg['lr'] = lr
        ema.tau = get_cosine_schedule(gs, total_steps, tc.ema_tau_start, tc.ema_tau_end)
        ep_loss += L_pred.item()

        if gs % 10 == 0:
            ps = preds.std(dim=[0,1]).mean().item()
            el = time.time()-start_time
            history.append((gs, L_pred.item(), L_sig.item(), ps))
            log(f'  step {gs:4d} | L_pred={L_pred.item():.4f}  L_sig={L_sig.item():.4f}  sig={ps:.3f}  lr={lr:.2e}  [{el:.0f}s]')
        gs += 1

    x_enc.eval(); pred.eval(); ema.apply()
    v_loss = 0.0; nv = 0
    with torch.no_grad():
        for x, labels in vl:
            x = x.to(device); x_m, x_orig, mm = apply_mask(x, cfg.mask)
            _, y_ints, _, _ = y_enc(x_orig, None)
            yt = {}
            for lidx in ce.target_layers:
                if lidx not in y_ints: continue
                tgt = y_ints[lidx]; mt = [tgt[b][mm[b]] for b in range(tgt.size(0))]
                max_m = max(mt[b].size(0) for b in range(len(mt)))
                if max_m == 0: continue
                pad = torch.zeros(x.size(0), max_m, tgt.size(-1), device=device)
                for b, m in enumerate(mt):
                    if m.size(0) > 0: pad[b,:m.size(0)] = m
                yt[lidx] = F.layer_norm(pad, (pad.size(-1),))
            x_out, _, _, _ = x_enc(x_m, mm)
            preds = pred(x_out, mm)
            for lidx, w in zip(ce.target_layers, ce.target_layer_weights):
                if lidx in yt:
                    yt_l = yt[lidx]; min_m = min(preds.size(1), yt_l.size(1))
                    v_loss += w*F.mse_loss(preds[:,:min_m], yt_l[:,:min_m]).item()
            nv += 1

    avg_tr = ep_loss/len(tl); avg_vl = v_loss/nv
    ep_t = time.time()-t_ep; el = time.time()-start_time
    eta = el/(epoch+1)*tc.epochs - el
    log(f'Epoch {epoch+1:2d} | train={avg_tr:.4f}  val={avg_vl:.4f}  [{ep_t:.0f}s]  elapsed={el:.0f}s  ETA={eta:.0f}s')

log(f'\nDone in {time.time()-start_time:.0f}s')
log(f'L_pred: {history[0][1]:.2f} -> {history[-1][1]:.2f}')
log(f'Best val: {min([h[1] for h in history]):.4f}')
