import os, re, gc, json, math, html, argparse, random
os.environ.setdefault('HF_HOME', '/kaggle/temp/hf')
try: os.makedirs(os.environ['HF_HOME'], exist_ok=True)
except Exception: pass
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

ROOT = os.environ.get('TA_ROOT', '/kaggle/working/tugas-akhir')
P = f'{ROOT}/data/processed'
LABELS = ['Negatif', 'Netral', 'Positif']; L2I = {l: i for i, l in enumerate(LABELS)}
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; USE_AMP = DEV == 'cuda'

p = argparse.ArgumentParser()
p.add_argument('--model', default='indobenchmark/indobert-base-p1')
p.add_argument('--tag', default='run1')
p.add_argument('--labels', default='final', choices=['final', 'before'])   # label latih: setelah / sebelum audit
p.add_argument('--drop_file', default='')                                  # CSV comment_id yang dibuang dari data LATIH saja
p.add_argument('--seed', type=int, default=42)
p.add_argument('--fold_seed', type=int, default=42)
p.add_argument('--folds', type=int, default=5)
p.add_argument('--epochs', type=int, default=4)
p.add_argument('--enc_lr', type=float, default=3e-5)
p.add_argument('--head_lr', type=float, default=1.2e-4)
p.add_argument('--layer_decay', type=float, default=0.92)
p.add_argument('--bs', type=int, default=32)
p.add_argument('--accum', type=int, default=1)
p.add_argument('--max_len', type=int, default=256)
p.add_argument('--wd', type=float, default=0.03)
p.add_argument('--drop', type=float, default=0.2)
p.add_argument('--ls', type=float, default=0.04)
p.add_argument('--eval_from', type=float, default=1.5)
p.add_argument('--swa_from', type=float, default=99.0)     # 99 = SWA mati; untuk Large pakai 2.5
p.add_argument('--evals_per_epoch', type=int, default=2)
p.add_argument('--grad_ckpt', action='store_true')
p.add_argument('--freeze_emb', action='store_true')
p.add_argument('--smoke', action='store_true')             # uji cepat: data kecil, 2 fold, 2 epoch
a = p.parse_args()
if a.smoke: a.folds, a.epochs, a.tag = 2, 2, a.tag + '_smoke'

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
seed_all(a.seed)

# ---------------- data ----------------
dev = pd.concat([pd.read_csv(f'{P}/splits/train_audited_v3.csv'), pd.read_csv(f'{P}/splits/validation_audited_v3.csv')], ignore_index=True)
te = pd.read_csv(f'{P}/splits/test.csv')
_aud = pd.read_csv(f'{P}/label_audit/test_audited_final.csv')[['comment_id', 'label_audited']].drop_duplicates('comment_id')
te = te.merge(_aud, on='comment_id', how='left'); te['label_audited'] = te['label_audited'].fillna(te['label_final'])
if a.smoke:
    dev = dev.sample(n=240, random_state=1).reset_index(drop=True); te = te.sample(n=60, random_state=1).reset_index(drop=True)
m_ = lambda s: s.map(L2I).values.astype(int)
y_final, y_before = m_(dev['label_final']), m_(dev['label_before_audit'])
y = y_final if a.labels == 'final' else y_before
yt, yt0 = m_(te['label_audited']), m_(te['label_final'])      # test: label audit (resmi) dan label asli

def clean_text(x):
    if pd.isna(x): return ''
    x = html.unescape(str(x)); x = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', x)
    x = re.sub(r'https?://\S+|www\.\S+', ' ', x, flags=re.I); x = re.sub(r'</?[a-zA-Z][^>]*>', ' ', x)
    x = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', x); return re.sub(r'\s+', ' ', x).strip()
_raw = pd.read_csv(f'{ROOT}/data/raw/comments_raw.csv', usecols=['comment_id', 'comment']).drop_duplicates('comment_id')
PMAP = {str(c): clean_text(x)[:140] for c, x in zip(_raw['comment_id'], _raw['comment'])}
def ctx(r):
    title = clean_text(r.get('video_title', ''))
    if str(r.get('is_reply', False)).lower() == 'true':
        pt_ = PMAP.get(str(r.get('parent_id', '')), '')
        return f'Balasan terhadap: {pt_} | Video: {title}' if pt_ else f'Balasan | Video: {title}'
    return f'Komentar Utama | Video: {title}'
tok = AutoTokenizer.from_pretrained(a.model); PAD = tok.pad_token_id
def encode(df):
    C = [ctx(r) for r in df.to_dict('records')]; T = df['comment_original'].fillna('').map(clean_text).tolist()
    return tok(C, T, truncation='only_second', max_length=a.max_len)['input_ids']
S_dev, S_te = encode(dev), encode(te)
print('token dev: median', int(np.median([len(s) for s in S_dev])), '| max', max(len(s) for s in S_dev), flush=True)

def batchify(seqs, idx):
    m = max(len(seqs[i]) for i in idx)
    ids = torch.full((len(idx), m), PAD, dtype=torch.long); att = torch.zeros((len(idx), m), dtype=torch.long)
    for r, i in enumerate(idx):
        s = seqs[i]; ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
    return ids, att

# ---------------- model ----------------
class Model(nn.Module):
    def __init__(s, name, n_last=4, drop=0.2):
        super().__init__()
        s.enc = AutoModel.from_pretrained(name, add_pooling_layer=False)
        H = s.enc.config.hidden_size; s.nl = n_last
        s.lw = nn.Parameter(torch.zeros(n_last))
        s.norm = nn.LayerNorm(2 * H); s.drop = nn.Dropout(drop)
        s.head = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, 3))
    def forward(s, ids, att):
        hs = s.enc(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[-s.nl:]
        w = torch.softmax(s.lw.float(), 0)
        x = sum(w[i] * hs[i].float() for i in range(s.nl))          # .float() SEBELUM masking
        m = att.unsqueeze(-1).float()
        mean = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = x.masked_fill(m == 0, -1e4).max(1).values
        z = s.norm(torch.cat([mean, mx], -1))
        return s.head(s.drop(z)).float()

def param_groups(model):
    nl = model.enc.config.num_hidden_layers; groups = {}
    nd = ('bias', 'LayerNorm.weight', 'LayerNorm.bias', 'norm.weight')
    for n, q in model.named_parameters():
        if not q.requires_grad: continue
        if n.startswith('enc.'):
            if '.encoder.layer.' in n:
                d = int(n.split('.encoder.layer.')[1].split('.')[0]); sc = a.layer_decay ** (nl - 1 - d)
            elif 'embeddings' in n: sc = a.layer_decay ** nl
            else: sc = 1.0
            lr = a.enc_lr * sc
        else: lr = a.head_lr
        wd = 0.0 if any(k in n for k in nd) else a.wd
        groups.setdefault((lr, wd), []).append(q)
    return [dict(params=v, lr=k[0], weight_decay=k[1]) for k, v in groups.items()]

def ac(): return torch.autocast(device_type=DEV, dtype=torch.float16, enabled=USE_AMP)

@torch.no_grad()
def predict(model, seqs):
    model.eval(); order = np.argsort([len(s) for s in seqs]); out = np.zeros((len(seqs), 3), np.float32)
    for k in range(0, len(order), 32):
        idx = order[k:k + 32]; ids, att = batchify(seqs, idx)
        with ac(): lg = model(ids.to(DEV), att.to(DEV))
        out[idx] = lg.float().cpu().numpy()
    return out

def swa_update(avg, model, n):
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            v = v.detach().float().cpu()
            avg[k] = v if k not in avg else avg[k] + (v - avg[k]) / (n + 1)

def run_fold(k, tr_idx, va_idx):
    seed_all(a.seed + k)
    model = Model(a.model, drop=a.drop).to(DEV)
    if a.grad_ckpt: model.enc.gradient_checkpointing_enable()
    if a.freeze_emb:
        for q in model.enc.embeddings.word_embeddings.parameters(): q.requires_grad = False
    opt = torch.optim.AdamW(param_groups(model), betas=(0.9, 0.98), eps=1e-6)
    n_micro = math.ceil(len(tr_idx) / a.bs); spe = math.ceil(n_micro / a.accum); total = spe * a.epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler(DEV, enabled=USE_AMP)
    eval_every = max(1, round(spe / a.evals_per_epoch))
    best = dict(f1=-1.0); avg = {}; n_swa = 0; step = 0
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(tr_idx)
        micro = [perm[i:i + a.bs] for i in range(0, len(perm), a.bs)]
        for mi, idx in enumerate(micro):
            ids, att = batchify(S_dev, idx); ids, att = ids.to(DEV), att.to(DEV); yy = torch.tensor(y[idx]).to(DEV)
            with ac(): lg = model(ids, att)
            loss = F.cross_entropy(lg.float(), yy, label_smoothing=a.ls)
            scaler.scale(loss / a.accum).backward()
            if (mi + 1) % a.accum == 0 or mi == len(micro) - 1:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step(); step += 1
                prog = step / spe
                if (step % eval_every == 0 and prog >= a.eval_from) or step == total:
                    vl = predict(model, [S_dev[i] for i in va_idx])
                    f = f1_score(y[va_idx], vl.argmax(1), average='macro')
                    if prog >= a.swa_from: swa_update(avg, model, n_swa); n_swa += 1
                    if f > best['f1']:
                        best = dict(f1=f, val=vl, test=predict(model, S_te), at=round(prog, 2), kind='best_ckpt')
                    print(f'  fold{k} ep{prog:.2f} val_macroF1={f:.4f} (best={best["f1"]:.4f}@{best["at"]})', flush=True)
                    model.train()
    if n_swa >= 2:
        model.load_state_dict(avg, strict=False)
        vl = predict(model, [S_dev[i] for i in va_idx]); f = f1_score(y[va_idx], vl.argmax(1), average='macro')
        print(f'  fold{k} SWA({n_swa} snapshot) val_macroF1={f:.4f}')
        if f >= best['f1']: best = dict(f1=f, val=vl, test=predict(model, S_te), at=-1, kind='swa')
    del model, opt, avg; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best

# ---------------- CV ----------------
out_dir = f'{ROOT}/results/{a.tag}'; os.makedirs(out_dir, exist_ok=True)
DROP = set(pd.read_csv(a.drop_file)['comment_id']) if a.drop_file else set()
splits = list(StratifiedKFold(a.folds, shuffle=True, random_state=a.fold_seed).split(np.zeros(len(y)), y))
for k, (tri, vai) in enumerate(splits):
    fp = f'{out_dir}/fold{k}.npz'
    if os.path.exists(fp): print('lewati fold', k, '(sudah selesai)'); continue
    if DROP:
        tri = np.array([i for i in tri if dev['comment_id'].iloc[i] not in DROP])
        print(f'fold{k}: data latih setelah filter = {len(tri)}', flush=True)
    b = run_fold(k, tri, vai)
    np.savez(fp, val_idx=vai, val=b['val'], test=b['test'], f1=b['f1'], kind=b['kind'])
    print(f'== fold{k} selesai F1={b["f1"]:.4f} ({b["kind"]})', flush=True)

fs = [f'{out_dir}/fold{k}.npz' for k in range(a.folds)]
if all(os.path.exists(f) for f in fs):
    oof = np.zeros((len(y), 3), np.float32); tl = []
    for f in fs:
        d = np.load(f); oof[d['val_idx']] = d['val']; tl.append(d['test'])
    tl = np.mean(tl, 0)
    np.save(f'{out_dir}/oof_logits.npy', oof); np.save(f'{out_dir}/test_logits.npy', tl)
    pr, pt = oof.argmax(1), tl.argmax(1)
    m = dict(oof_acc_vs_final=accuracy_score(y_final, pr), oof_f1_vs_final=f1_score(y_final, pr, average='macro'),
             oof_acc_vs_before=accuracy_score(y_before, pr), oof_f1_vs_before=f1_score(y_before, pr, average='macro'),
             test_acc_audited=accuracy_score(yt, pt), test_f1_audited=f1_score(yt, pt, average='macro'),
             test_acc_original=accuracy_score(yt0, pt), test_f1_original=f1_score(yt0, pt, average='macro'), args=vars(a))
    json.dump(m, open(f'{out_dir}/metrics.json', 'w'), indent=2, default=str)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != 'args'}, indent=2))
