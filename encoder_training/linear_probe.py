"""Linear probing: freeze trained x-encoder, train classification head with cross-entropy.

Phase 1: Pretrain encoder (same as quick_train, save checkpoint)
Phase 2: Freeze x-encoder, add linear head, train on activity labels, evaluate.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn, torch.nn.functional as F, time, copy
from torch.utils.data import DataLoader
from config import Config
from data.dataset import WISDMWindowDataset
from data.augment import apply_mask
from models.encoder import MaskedEncoder
from models.predictor import Predictor, SigRegHead
from train import EMAModel, get_cosine_schedule

def log(*args):
    print(*args, flush=True)

# ---- Config ----
cfg = Config()
cfg.data.train_subjects = [1600,1601,1602,1603,1604,1605]
cfg.data.val_subjects   = [1606,1607]
cfg.data.test_subjects  = [1608,1609]
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

device = 'cuda'; ce = cfg.encoder; tc = cfg.train

# ============================================================
# Phase 1: Pretrain Encoder
# ============================================================
log('='*60)
log('Phase 1: Self-Supervised Pretraining')
log('='*60)

def collate_ssl(b):
    x = torch.stack([it['x'] for it in b])
    return x, torch.tensor([it['label_idx'] for it in b])

def collate_sup(b):
    x = torch.stack([it['x'] for it in b])
    y = torch.tensor([it['label_idx'] for it in b], dtype=torch.long)
    return x, y

log('Loading data...')
t0 = time.time()
train_ds = WISDMWindowDataset(os.path.join(cfg.data.dataset_root, cfg.data.raw_dir),
    cfg.data.train_subjects, cfg.data.window_size, cfg.data.window_stride, cfg.data.sensors)
val_ds = WISDMWindowDataset(os.path.join(cfg.data.dataset_root, cfg.data.raw_dir),
    cfg.data.val_subjects, cfg.data.window_size, cfg.data.window_stride, cfg.data.sensors)
test_ds = WISDMWindowDataset(os.path.join(cfg.data.dataset_root, cfg.data.raw_dir),
    cfg.data.test_subjects, cfg.data.window_size, cfg.data.window_stride, cfg.data.sensors)
log(f'Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}  ({time.time()-t0:.1f}s)')

tl = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True, collate_fn=collate_ssl, drop_last=True)
vl = DataLoader(val_ds, batch_size=tc.batch_size, shuffle=False, collate_fn=collate_ssl, drop_last=False)

x_enc = MaskedEncoder(ce).to(device)
y_enc = MaskedEncoder(ce).to(device)
pred = Predictor(ce.embed_dim, ce.predictor_n_layers, ce.n_heads, ce.predictor_mlp_hidden).to(device)
sigreg = SigRegHead(ce.embed_dim).to(device)
y_enc.load_state_dict(x_enc.state_dict())
ema = EMAModel(y_enc, tau=tc.ema_tau_start)

params = list(x_enc.parameters()) + list(pred.parameters()) + list(sigreg.parameters())
log(f'Encoder params: {sum(p.numel() for p in params)/1e6:.1f}M')
opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)

total_steps = tc.epochs * len(tl)
warmup_steps = tc.warmup_epochs * len(tl)

gs = 0; start = time.time()
for epoch in range(tc.epochs):
    x_enc.train(); pred.train(); y_enc.eval()
    ep_loss = 0.0
    for x, _ in tl:
        x = x.to(device)
        x_m, x_orig, mm = apply_mask(x, cfg.mask)
        with torch.no_grad():
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
        x_out, _, cls_out, _ = x_enc(x_m, mm)
        preds = pred(x_out, mm)
        L_pred = torch.tensor(0.0, device=device)
        for lidx, w in zip(ce.target_layers, ce.target_layer_weights):
            if lidx in yt:
                yt_l = yt[lidx]; min_m = min(preds.size(1), yt_l.size(1))
                L_pred = L_pred + w*F.mse_loss(preds[:,:min_m], yt_l[:,:min_m])
        z = sigreg(preds); L_sig = sigreg.compute_loss(z, tc.lambda_var, tc.lambda_ent)
        (L_pred+L_sig).backward(); opt.step(); opt.zero_grad()
        with torch.no_grad(): ema.update(x_enc)
        if gs < warmup_steps: lr = tc.lr*(gs+1)/warmup_steps
        else: lr = get_cosine_schedule(gs-warmup_steps, total_steps-warmup_steps, tc.lr, tc.min_lr)
        for pg in opt.param_groups: pg['lr'] = lr
        ema.tau = get_cosine_schedule(gs, total_steps, tc.ema_tau_start, tc.ema_tau_end)
        ep_loss += L_pred.item(); gs += 1

    log(f'  SSL Epoch {epoch+1:2d} | L_pred={ep_loss/len(tl):.4f}  [{time.time()-start:.0f}s]')

log(f'Pretraining done in {time.time()-start:.0f}s')

# ============================================================
# Phase 2: Linear Probing
# ============================================================
log('')
log('='*60)
log('Phase 2: Linear Probing (Frozen Encoder + Classifier Head)')
log('='*60)

# Build supervised loaders
train_sup = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True, collate_fn=collate_sup, drop_last=True)
val_sup = DataLoader(val_ds, batch_size=tc.batch_size, shuffle=False, collate_fn=collate_sup, drop_last=False)
test_sup = DataLoader(test_ds, batch_size=tc.batch_size, shuffle=False, collate_fn=collate_sup, drop_last=False)

# Freeze x-encoder
for p in x_enc.parameters():
    p.requires_grad = False
x_enc.eval()

# Classification head
n_classes = 18
classifier = nn.Sequential(
    nn.Linear(ce.embed_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, n_classes),
).to(device)

log(f'Classifier params: {sum(p.numel() for p in classifier.parameters()):,}')

opt_cls = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cls, T_max=30)

best_val_acc = 0.0; best_cls_state = None; start_cls = time.time()

for epoch in range(30):
    classifier.train()
    tr_loss = 0.0; tr_correct = 0; tr_total = 0
    for x, y in train_sup:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, _, cls_out, _ = x_enc(x, mask_matrix=None)  # full signal, no mask
        logits = classifier(cls_out)
        loss = F.cross_entropy(logits, y)
        opt_cls.zero_grad(); loss.backward(); opt_cls.step()
        tr_loss += loss.item(); tr_correct += (logits.argmax(1)==y).sum().item(); tr_total += y.size(0)
    scheduler.step()

    # Validation
    classifier.eval()
    val_correct = 0; val_total = 0; val_loss = 0.0
    all_preds = []; all_labels = []
    with torch.no_grad():
        for x, y in val_sup:
            x, y = x.to(device), y.to(device)
            _, _, cls_out, _ = x_enc(x, mask_matrix=None)
            logits = classifier(cls_out)
            val_loss += F.cross_entropy(logits, y).item()
            val_correct += (logits.argmax(1)==y).sum().item(); val_total += y.size(0)
            all_preds.append(logits.argmax(1).cpu()); all_labels.append(y.cpu())

    val_acc = val_correct/val_total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_cls_state = copy.deepcopy(classifier.state_dict())

    if epoch % 3 == 0:
        log(f'  Probe Epoch {epoch:2d} | train_loss={tr_loss/len(train_sup):.4f}  train_acc={tr_correct/tr_total:.3f}  val_acc={val_acc:.3f}')

# ---- Final Test Evaluation ----
classifier.load_state_dict(best_cls_state)
classifier.eval()

test_correct = 0; test_total = 0
all_preds = []; all_labels = []
with torch.no_grad():
    for x, y in test_sup:
        x, y = x.to(device), y.to(device)
        _, _, cls_out, _ = x_enc(x, mask_matrix=None)
        logits = classifier(cls_out)
        preds = logits.argmax(1)
        test_correct += (preds==y).sum().item(); test_total += y.size(0)
        all_preds.append(preds.cpu()); all_labels.append(y.cpu())

test_acc = test_correct / test_total
all_preds = torch.cat(all_preds); all_labels = torch.cat(all_labels)

# Per-class accuracy
from collections import Counter
per_class_correct = Counter(); per_class_total = Counter()
for p, l in zip(all_preds.tolist(), all_labels.tolist()):
    per_class_total[l] += 1
    if p == l: per_class_correct[l] += 1

act_names = {0:'A-walk',1:'B-jog',2:'C-stairs',3:'D-sit',4:'E-stand',5:'F-type',6:'G-teeth',
             7:'H-soup',8:'I-chips',9:'J-pasta',10:'K-drink',11:'L-sandw',12:'M-kick',
             13:'O-catch',14:'P-dribble',15:'Q-write',16:'R-clap',17:'S-fold'}

log(f'\n{"="*60}')
log(f'PHASE 2 RESULTS (Linear Probe, Frozen Encoder)')
log(f'{"="*60}')
log(f'Random baseline: {1/n_classes:.1%}')
log(f'Test Accuracy: {test_acc:.2%} ({test_correct}/{test_total})')
log(f'Best Val Accuracy: {best_val_acc:.2%}')
log(f'\nPer-class accuracy:')
for cls_idx in sorted(per_class_total.keys()):
    acc = per_class_correct[cls_idx]/per_class_total[cls_idx]
    name = act_names.get(cls_idx, '?')
    bar = '#' * int(acc*40)
    log(f'  {name:>12s}: {acc:.2%} {bar}')
log(f'\nTotal time: {time.time()-start:.0f}s')
