"""
Solution V2.2: Arabic+English ABSA — Enhanced (XLM-R-large backbone)
=====================================================================

V1    OOF = 0.6161 (XLM-R-base baseline)
V2    = mDeBERTa-v3-base -> NaN instability, dead run
V2.1  = XLM-R-large, all V2 loss/fallback improvements (good loss curves but
        predicted on wrong test file: DeepX_unlabeled.xlsx)
V2.2  = V2.1 with TEST_PATH corrected to DeepX_hidden_test.xlsx (the graded set)

No other changes from V2.1. Expected OOF: ~0.67-0.73.

Changes from V1 (all V2 improvements + stable backbone):

1. BACKBONE: xlm-roberta-large (was xlm-roberta-base in V1, mDeBERTa in V2).
   560M params vs 278M -- a real capacity upgrade. Same family as V1 so stability
   is proven. LR reduced to 1e-5 and warmup extended to 15% per large-model convention.
   Batch size 8 with grad_accum 2 for effective batch 16 with memory headroom.

2. CLASS-WEIGHTED LOSSES (core fix for V1's zero-firing rare classes):
   - Aspect BCE: pos_weight per class = sqrt((N-pos)/pos). Gives 'none' weight 5.8x,
     'cleanliness' 3.1x, 'delivery' 3.4x vs 'service' 1.0x. Sqrt-scaling is a middle
     ground — raw pos_weight would overshoot (none=34x) and destabilize training.
   - Sentiment CE: balanced-sqrt class weights, normalized to mean 1.
     'neutral' gets weight 1.86x, 'positive'/'negative' get ~0.57x.

3. MORE TRAINING: 6 epochs (was 4). Loss was still decreasing at 4.

4. LONGER INPUT: max_len 256 (was 192). Catches more of the long tail (p99 train ~350 tokens).

5. MULTI-SEED ENSEMBLE: 2 seeds per fold -> 10 models instead of 5.
   Same fold splits (deterministic) with different init/shuffle seeds. Averaged within fold.

6. SMARTER FALLBACK: when no aspect crosses threshold, predict argmax aspect (was: always 'general').
   Reduces 'general' over-prediction. Sentiment for all fallback cases uses star-rating oracle
   (model confidence is implicitly low if no aspect crossed threshold).

7. JOINT OPTIMIZATION: coordinate ascent tunes both per-aspect thresholds AND neutral_min.

8. DIAGNOSTICS: per-aspect OOF F1, aspect-only F1, sentiment-given-aspect accuracy.
   Makes V2-vs-V1 comparison attributable.

Unchanged from V1:
  - Dual-head architecture (aspect sigmoid + 9x3 sentiment softmax)
  - Metadata-in-text input format
  - Mean pooling, dropout 0.1
  - AdamW, linear warmup, fp32, TF32 off, deterministic cuDNN
  - 5-fold stratified CV on (num_aspects, stars, script)
  - 'none' exclusivity post-processing

New in V2.1: NaN guard in training loop (skips step if loss is NaN/Inf).
Defensive — XLM-R-large is stable, but costs nothing to have.

Runtime: ~60-80 min on L40 (xlm-r-large is ~2x slower than base per step).
"""

import os
import json
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# CONFIG
# =============================================================================
SEED = 42
MODEL_NAME = "xlm-roberta-large"
MAX_LEN = 256
BATCH_SIZE = 8           # halved for large model memory headroom
EVAL_BATCH_SIZE = 16
LR = 1e-5                # reduced for large model (was 2e-5 on base)
EPOCHS = 6
WARMUP_RATIO = 0.15      # extended from 0.10 for large model stability
WEIGHT_DECAY = 0.01
N_FOLDS = 5
SEEDS_PER_FOLD = 2       # 2 random inits per fold -> 10 models total
GRAD_ACCUM = 2           # effective batch = BATCH_SIZE * GRAD_ACCUM = 16
SENT_LOSS_WEIGHT = 1.0
POS_WEIGHT_METHOD = 'sqrt'          # 'sqrt' | 'raw' | 'none'
SENT_WEIGHT_METHOD = 'balanced_sqrt' # 'balanced_sqrt' | 'balanced' | 'none'
INITIAL_NEUTRAL_MIN = 0.50

ASPECTS = ['food', 'service', 'price', 'cleanliness', 'delivery',
           'ambiance', 'app_experience', 'general', 'none']
SENTIMENTS = ['positive', 'negative', 'neutral']
N_ASPECTS = len(ASPECTS)
N_SENTIMENTS = len(SENTIMENTS)
ASPECT_TO_IDX = {a: i for i, a in enumerate(ASPECTS)}
SENT_TO_IDX = {s: i for i, s in enumerate(SENTIMENTS)}

TRAIN_PATH = "DeepX_train.xlsx"
VAL_PATH = "DeepX_validation.xlsx"
TEST_PATH = "DeepX_hidden_test.xlsx"   # v2.2 fix: was DeepX_unlabeled.xlsx (wrong target — that's an unlabeled pool, not the graded set)
OUTPUT_DIR = "./working"
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.json")

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# =============================================================================
# DATA
# =============================================================================
def load_df(path):
    return pd.read_excel(path)

def parse_labels(row):
    aspects = json.loads(row['aspects'])
    sents = json.loads(row['aspect_sentiments'])
    aspect_vec = np.zeros(N_ASPECTS, dtype=np.float32)
    sent_mat = np.zeros((N_ASPECTS, N_SENTIMENTS), dtype=np.float32)
    for a in aspects:
        if a not in ASPECT_TO_IDX: continue
        ai = ASPECT_TO_IDX[a]
        aspect_vec[ai] = 1.0
        s = sents.get(a, 'neutral')
        si = SENT_TO_IDX.get(s, SENT_TO_IDX['neutral'])
        sent_mat[ai, si] = 1.0
    return aspect_vec, sent_mat

def script_of(s):
    s = str(s)
    ar = sum(1 for c in s if '\u0600' <= c <= '\u06ff')
    en = sum(1 for c in s if c.isascii() and c.isalpha())
    if ar == 0 and en == 0: return 'other'
    if ar > 3 * en: return 'ar'
    if en > 3 * ar: return 'en'
    return 'mix'

def format_input(row):
    stars = int(row['star_rating'])
    cat = str(row['business_category'])[:40]
    plat = str(row['platform'])[:20]
    text = str(row['review_text'])
    return f"[STARS={stars}] [CAT={cat}] [PLAT={plat}] {text}"

def compute_aspect_pos_weight(train_df, method='sqrt'):
    """Return (N_ASPECTS,) pos_weight for BCEWithLogitsLoss."""
    N = len(train_df)
    counts = np.zeros(N_ASPECTS)
    for _, r in train_df.iterrows():
        for a in json.loads(r['aspects']):
            if a in ASPECT_TO_IDX:
                counts[ASPECT_TO_IDX[a]] += 1
    raw = (N - counts) / np.maximum(counts, 1)
    if method == 'sqrt':
        return np.sqrt(raw).astype(np.float32)
    elif method == 'raw':
        return raw.astype(np.float32)
    else:
        return np.ones(N_ASPECTS, dtype=np.float32)

def compute_sent_class_weight(train_df, method='balanced_sqrt'):
    """Return (N_SENTIMENTS,) class weights for sentiment CE, normalized to mean 1."""
    counts = np.zeros(N_SENTIMENTS)
    for _, r in train_df.iterrows():
        for s in json.loads(r['aspect_sentiments']).values():
            if s in SENT_TO_IDX:
                counts[SENT_TO_IDX[s]] += 1
    total = counts.sum()
    w = total / (N_SENTIMENTS * np.maximum(counts, 1))
    if method == 'balanced_sqrt':
        w = np.sqrt(w)
    elif method == 'balanced':
        pass
    else:
        w = np.ones(N_SENTIMENTS)
    w = w / w.mean()
    return w.astype(np.float32)

# =============================================================================
# DATASET
# =============================================================================
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN, has_labels=True):
        self.texts = [format_input(r) for _, r in df.iterrows()]
        self.ids = df['review_id'].astype(int).tolist()
        self.stars = df['star_rating'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_labels = has_labels
        if has_labels:
            avs, sms = [], []
            for _, r in df.iterrows():
                av, sm = parse_labels(r)
                avs.append(av); sms.append(sm)
            self.aspect_vecs = np.stack(avs)
            self.sent_mats = np.stack(sms)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'review_id': self.ids[idx],
            'stars': self.stars[idx],
        }
        if self.has_labels:
            item['aspect_targets'] = torch.tensor(self.aspect_vecs[idx], dtype=torch.float32)
            item['sentiment_targets'] = torch.tensor(self.sent_mats[idx], dtype=torch.float32)
        return item

# =============================================================================
# MODEL
# =============================================================================
class ABSAModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, n_aspects=N_ASPECTS,
                 n_sentiments=N_SENTIMENTS, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.aspect_head = nn.Linear(H, n_aspects)
        self.sentiment_head = nn.Linear(H, n_aspects * n_sentiments)
        self.n_aspects = n_aspects
        self.n_sentiments = n_sentiments

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = self.dropout(pooled)
        aspect_logits = self.aspect_head(pooled)
        sent_logits = self.sentiment_head(pooled).view(-1, self.n_aspects, self.n_sentiments)
        return aspect_logits, sent_logits

# =============================================================================
# LOSS (class-weighted)
# =============================================================================
def compute_loss(aspect_logits, sent_logits, aspect_targets, sent_targets,
                 aspect_pos_weight=None, sent_class_weight=None,
                 sent_loss_weight=SENT_LOSS_WEIGHT):
    if aspect_pos_weight is not None:
        bce = nn.functional.binary_cross_entropy_with_logits(
            aspect_logits, aspect_targets, pos_weight=aspect_pos_weight)
    else:
        bce = nn.functional.binary_cross_entropy_with_logits(aspect_logits, aspect_targets)

    # Sentiment: weighted CE masked to present aspects
    log_probs = torch.log_softmax(sent_logits, dim=-1)  # (B, A, S)
    if sent_class_weight is not None:
        # sent_targets is one-hot; weight the correct class by its class weight
        weighted = sent_targets * sent_class_weight.view(1, 1, -1)
        per_aspect_ce = -(weighted * log_probs).sum(dim=-1)
    else:
        per_aspect_ce = -(sent_targets * log_probs).sum(dim=-1)
    present = aspect_targets
    denom = present.sum().clamp(min=1.0)
    sent_ce = (per_aspect_ce * present).sum() / denom
    return bce + sent_loss_weight * sent_ce, bce.item(), sent_ce.item()

# =============================================================================
# METRICS
# =============================================================================
def row_to_tuples(aspects_list, sent_dict):
    return {(a, sent_dict.get(a, 'neutral')) for a in aspects_list}

def tuple_micro_f1(gold_rows, pred_rows):
    tp = fp = fn = 0
    for g, p in zip(gold_rows, pred_rows):
        g_set = row_to_tuples(g['aspects'], g['aspect_sentiments'])
        p_set = row_to_tuples(p['aspects'], p['aspect_sentiments'])
        tp += len(g_set & p_set); fp += len(p_set - g_set); fn += len(g_set - p_set)
    if tp == 0: return 0.0
    pr = tp / (tp + fp); rc = tp / (tp + fn)
    return 2 * pr * rc / (pr + rc)

def aspect_only_f1(gold_rows, pred_rows):
    tp = fp = fn = 0
    for g, p in zip(gold_rows, pred_rows):
        g_set = set(g['aspects']); p_set = set(p['aspects'])
        tp += len(g_set & p_set); fp += len(p_set - g_set); fn += len(g_set - p_set)
    if tp == 0: return 0.0
    pr = tp / (tp + fp); rc = tp / (tp + fn)
    return 2 * pr * rc / (pr + rc)

def sent_given_aspect_acc(gold_rows, pred_rows):
    correct = total = 0
    for g, p in zip(gold_rows, pred_rows):
        common = set(g['aspects']) & set(p['aspects'])
        for a in common:
            total += 1
            if g['aspect_sentiments'].get(a) == p['aspect_sentiments'].get(a):
                correct += 1
    return correct / max(total, 1), total

def per_aspect_f1(gold_rows, pred_rows):
    stats = {a: {'tp': 0, 'fp': 0, 'fn': 0} for a in ASPECTS}
    for g, p in zip(gold_rows, pred_rows):
        g_tuples = {(a, g['aspect_sentiments'].get(a, 'neutral')) for a in g['aspects']}
        p_tuples = {(a, p['aspect_sentiments'].get(a, 'neutral')) for a in p['aspects']}
        for (a, _) in g_tuples & p_tuples: stats[a]['tp'] += 1
        for (a, _) in p_tuples - g_tuples: stats[a]['fp'] += 1
        for (a, _) in g_tuples - p_tuples: stats[a]['fn'] += 1
    result = {}
    for a, st in stats.items():
        tp, fp, fn = st['tp'], st['fp'], st['fn']
        if tp == 0:
            f1 = p = r = 0.0
        else:
            p = tp / (tp + fp); r = tp / (tp + fn); f1 = 2 * p * r / (p + r)
        result[a] = {'f1': f1, 'precision': p, 'recall': r, 'tp': tp, 'fp': fp, 'fn': fn}
    return result

# =============================================================================
# DECODE (smart fallback)
# =============================================================================
def decode_single(aspect_probs, sent_probs, stars, thresholds, neutral_min):
    selected = [i for i in range(N_ASPECTS) if aspect_probs[i] >= thresholds[i]]
    none_idx = ASPECT_TO_IDX['none']
    if none_idx in selected and len(selected) > 1:
        selected = [i for i in selected if i != none_idx]
    fallback = False
    if len(selected) == 0:
        # NEW in V2: take argmax aspect, not always 'general'
        selected = [int(np.argmax(aspect_probs))]
        fallback = True

    aspects_list = [ASPECTS[i] for i in selected]
    sent_dict = {}
    neu_i, pos_i, neg_i = SENT_TO_IDX['neutral'], SENT_TO_IDX['positive'], SENT_TO_IDX['negative']
    for i in selected:
        asp = ASPECTS[i]
        if asp == 'none':
            sent_dict[asp] = 'neutral'
            continue
        if fallback:
            # Star-rating oracle (low model confidence -> trust stars)
            if stars >= 4: sent_dict[asp] = 'positive'
            elif stars <= 2: sent_dict[asp] = 'negative'
            else: sent_dict[asp] = 'neutral'
            continue
        probs = sent_probs[i]
        argmax = int(np.argmax(probs))
        if argmax == neu_i and probs[neu_i] < neutral_min:
            argmax = pos_i if probs[pos_i] >= probs[neg_i] else neg_i
        sent_dict[asp] = SENTIMENTS[argmax]
    return aspects_list, sent_dict

def decode_all(aspect_probs, sent_probs, stars_arr, thresholds, neutral_min):
    return [
        {'aspects': a, 'aspect_sentiments': s}
        for i in range(len(aspect_probs))
        for a, s in [decode_single(aspect_probs[i], sent_probs[i], stars_arr[i],
                                   thresholds, neutral_min)]
    ]

# =============================================================================
# THRESHOLD TUNING (joint: aspect thresholds + neutral_min)
# =============================================================================
def tune_thresholds(aspect_probs, sent_probs, stars_arr, gold_rows,
                    aspect_grid=None, neutral_grid=None, passes=2):
    if aspect_grid is None:
        aspect_grid = np.arange(0.10, 0.81, 0.05)
    if neutral_grid is None:
        neutral_grid = np.arange(0.30, 0.81, 0.05)

    thresholds = np.full(N_ASPECTS, 0.5, dtype=np.float32)
    neutral_min = float(INITIAL_NEUTRAL_MIN)

    pr = decode_all(aspect_probs, sent_probs, stars_arr, thresholds, neutral_min)
    best = tuple_micro_f1(gold_rows, pr)
    print(f"  init (thr=0.5, neutral_min={neutral_min:.2f}) -> F1={best:.4f}")

    for p in range(passes):
        # Aspect thresholds
        for ai in range(N_ASPECTS):
            best_t = float(thresholds[ai])
            for t in aspect_grid:
                thresholds[ai] = t
                ppr = decode_all(aspect_probs, sent_probs, stars_arr, thresholds, neutral_min)
                f1 = tuple_micro_f1(gold_rows, ppr)
                if f1 > best:
                    best = f1; best_t = float(t)
            thresholds[ai] = best_t
        # Neutral_min
        for nm in neutral_grid:
            ppr = decode_all(aspect_probs, sent_probs, stars_arr, thresholds, float(nm))
            f1 = tuple_micro_f1(gold_rows, ppr)
            if f1 > best:
                best = f1; neutral_min = float(nm)
        print(f"  pass {p+1}: F1={best:.4f}  neutral_min={neutral_min:.2f}  "
              f"thresholds={[round(float(x),2) for x in thresholds]}")
    return thresholds, neutral_min, best

# =============================================================================
# TRAINING
# =============================================================================
def train_one_model(tr_df, vl_df, tokenizer, device, fold, seed,
                    aspect_pos_weight_tensor, sent_class_weight_tensor):
    seed_everything(seed)
    train_ds = ABSADataset(tr_df, tokenizer, has_labels=True)
    valid_ds = ABSADataset(vl_df, tokenizer, has_labels=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=2)

    model = ABSAModel().to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=LR)
    total_steps = max(1, len(train_loader) * EPOCHS // GRAD_ACCUM)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0; count = 0; n_skipped = 0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            at = batch['aspect_targets'].to(device)
            st = batch['sentiment_targets'].to(device)
            a_log, s_log = model(input_ids, attn)
            loss, _, _ = compute_loss(
                a_log, s_log, at, st,
                aspect_pos_weight=aspect_pos_weight_tensor,
                sent_class_weight=sent_class_weight_tensor,
            )
            loss = loss / GRAD_ACCUM
            # NaN guard: skip step if loss is invalid (defensive — shouldn't fire with XLM-R)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                n_skipped += 1
                continue
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            running += loss.item() * GRAD_ACCUM
            count += 1
        skip_note = f"  skipped={n_skipped}" if n_skipped > 0 else ""
        print(f"  Fold{fold} Seed{seed} Ep{epoch+1}/{EPOCHS}: "
              f"loss={running/max(count,1):.4f}{skip_note}  ({time.time()-t0:.1f}s)")

    # Predict on held-out fold
    model.eval()
    all_a, all_s = [], []
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            a_log, s_log = model(input_ids, attn)
            all_a.append(torch.sigmoid(a_log).cpu().numpy())
            all_s.append(torch.softmax(s_log, dim=-1).cpu().numpy())
    return model, np.concatenate(all_a), np.concatenate(all_s)

@torch.no_grad()
def predict_with_model(model, df, tokenizer, device):
    ds = ABSADataset(df, tokenizer, has_labels=False)
    loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=2)
    all_a, all_s = [], []
    model.eval()
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        a_log, s_log = model(input_ids, attn)
        all_a.append(torch.sigmoid(a_log).cpu().numpy())
        all_s.append(torch.softmax(s_log, dim=-1).cpu().numpy())
    return np.concatenate(all_a), np.concatenate(all_s)

# =============================================================================
# FOLDS
# =============================================================================
def make_folds(df, n_folds=N_FOLDS, seed=SEED):
    strata = []
    for _, r in df.iterrows():
        try:
            n_a = min(len(json.loads(r['aspects'])), 4)
        except Exception:
            n_a = 1
        strata.append(f"{n_a}_{int(r['star_rating'])}_{script_of(r['review_text'])}")
    strata = pd.Series(strata)
    rare = set(strata.value_counts()[lambda x: x < n_folds].index)
    strata = strata.apply(lambda x: '__rare__' if x in rare else x)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(df, strata))

def df_to_gold_rows(df):
    return [
        {'aspects': list(json.loads(r['aspects'])),
         'aspect_sentiments': dict(json.loads(r['aspect_sentiments']))}
        for _, r in df.iterrows()
    ]

# =============================================================================
# MAIN
# =============================================================================
def main():
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}  max_len={MAX_LEN}  bs={BATCH_SIZE}  "
          f"epochs={EPOCHS}  folds={N_FOLDS}  seeds/fold={SEEDS_PER_FOLD}")

    train_df = load_df(TRAIN_PATH).reset_index(drop=True)
    val_df = load_df(VAL_PATH).reset_index(drop=True)
    test_df = load_df(TEST_PATH).reset_index(drop=True)
    print(f"Sizes: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # Class weights
    pw = compute_aspect_pos_weight(train_df, method=POS_WEIGHT_METHOD)
    sw = compute_sent_class_weight(train_df, method=SENT_WEIGHT_METHOD)
    print(f"Aspect pos_weight ({POS_WEIGHT_METHOD}):")
    for a, w in zip(ASPECTS, pw):
        print(f"  {a:18s}: {w:.3f}")
    print(f"Sent class weights ({SENT_WEIGHT_METHOD}): "
          f"{dict(zip(SENTIMENTS, [round(float(x),3) for x in sw]))}")
    pw_t = torch.tensor(pw, dtype=torch.float32).to(device)
    sw_t = torch.tensor(sw, dtype=torch.float32).to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    folds = make_folds(train_df)

    # Containers
    oof_a = np.zeros((len(train_df), N_ASPECTS), dtype=np.float32)
    oof_s = np.zeros((len(train_df), N_ASPECTS, N_SENTIMENTS), dtype=np.float32)
    val_a = np.zeros((len(val_df), N_ASPECTS), dtype=np.float32)
    val_s = np.zeros((len(val_df), N_ASPECTS, N_SENTIMENTS), dtype=np.float32)
    test_a = np.zeros((len(test_df), N_ASPECTS), dtype=np.float32)
    test_s = np.zeros((len(test_df), N_ASPECTS, N_SENTIMENTS), dtype=np.float32)

    total_models = N_FOLDS * SEEDS_PER_FOLD

    for fold, (tr_idx, vl_idx) in enumerate(folds):
        print(f"\n===== FOLD {fold+1}/{N_FOLDS} =====")
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        vl_df = train_df.iloc[vl_idx].reset_index(drop=True)

        # Accumulate within-fold across seeds
        fold_v_a = np.zeros((len(vl_idx), N_ASPECTS), dtype=np.float32)
        fold_v_s = np.zeros((len(vl_idx), N_ASPECTS, N_SENTIMENTS), dtype=np.float32)

        for seed_idx in range(SEEDS_PER_FOLD):
            seed = SEED + fold * 1000 + seed_idx * 100000
            model, v_a, v_s = train_one_model(
                tr_df, vl_df, tokenizer, device, fold, seed, pw_t, sw_t)
            fold_v_a += v_a / SEEDS_PER_FOLD
            fold_v_s += v_s / SEEDS_PER_FOLD

            # Predict on val + test; average over all (fold, seed) pairs = total_models
            ta, ts = predict_with_model(model, test_df, tokenizer, device)
            va, vs = predict_with_model(model, val_df, tokenizer, device)
            test_a += ta / total_models
            test_s += ts / total_models
            val_a += va / total_models
            val_s += vs / total_models

            del model
            torch.cuda.empty_cache()

        oof_a[vl_idx] = fold_v_a
        oof_s[vl_idx] = fold_v_s

    # Save probabilities
    np.savez(os.path.join(OUTPUT_DIR, 'probs_v2_2.npz'),
             oof_a=oof_a, oof_s=oof_s,
             val_a=val_a, val_s=val_s,
             test_a=test_a, test_s=test_s,
             train_ids=train_df['review_id'].astype(int).values,
             val_ids=val_df['review_id'].astype(int).values,
             test_ids=test_df['review_id'].astype(int).values)

    # Tune thresholds + neutral_min jointly
    print("\n===== JOINT THRESHOLD TUNING (OOF) =====")
    gold_train = df_to_gold_rows(train_df)
    stars_train = train_df['star_rating'].astype(int).values
    thresholds, neutral_min, oof_f1 = tune_thresholds(
        oof_a, oof_s, stars_train, gold_train)
    print(f"==> OOF tuple micro F1 (tuned): {oof_f1:.4f}")

    # Diagnostics: OOF
    oof_pred = decode_all(oof_a, oof_s, stars_train, thresholds, neutral_min)
    a_only_f1 = aspect_only_f1(gold_train, oof_pred)
    s_acc, n_common = sent_given_aspect_acc(gold_train, oof_pred)
    print(f"    aspect-only F1:       {a_only_f1:.4f}")
    print(f"    sent | correct asp:   {s_acc:.4f}  (n={n_common})")
    print(f"    per-aspect OOF F1:")
    pa = per_aspect_f1(gold_train, oof_pred)
    for a in ASPECTS:
        st = pa[a]
        print(f"      {a:18s}: F1={st['f1']:.3f}  P={st['precision']:.3f}  "
              f"R={st['recall']:.3f}  tp={st['tp']} fp={st['fp']} fn={st['fn']}")

    # Given val
    gold_val = df_to_gold_rows(val_df)
    stars_val = val_df['star_rating'].astype(int).values
    val_pred = decode_all(val_a, val_s, stars_val, thresholds, neutral_min)
    val_f1 = tuple_micro_f1(gold_val, val_pred)
    print(f"\n==> Val tuple micro F1: {val_f1:.4f}  (distribution matches train, not test)")

    # Test submission
    stars_test = test_df['star_rating'].astype(int).values
    test_pred = decode_all(test_a, test_s, stars_test, thresholds, neutral_min)
    submission = [
        {'review_id': int(rid),
         'aspects': list(pr['aspects']),
         'aspect_sentiments': {k: v for k, v in pr['aspect_sentiments'].items()}}
        for rid, pr in zip(test_df['review_id'].astype(int).tolist(), test_pred)
    ]
    with open(SUBMISSION_PATH, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(submission)} predictions -> {SUBMISSION_PATH}")

    # Prediction diagnostics
    aspect_ct = Counter(); sent_ct = Counter(); n_asp = []
    for pr in test_pred:
        n_asp.append(len(pr['aspects']))
        for a in pr['aspects']:
            aspect_ct[a] += 1
            sent_ct[pr['aspect_sentiments'][a]] += 1
    print("\nTest prediction diagnostics:")
    print(f"  avg aspects per review: {np.mean(n_asp):.2f}")
    print(f"  #aspects distribution: {dict(Counter(n_asp))}")
    print(f"  aspect usage: {dict(aspect_ct.most_common())}")
    print(f"  sentiment usage: {dict(sent_ct.most_common())}")

    # Save metadata
    with open(os.path.join(OUTPUT_DIR, 'thresholds_v2_2.json'), 'w') as f:
        json.dump({
            'aspects': ASPECTS,
            'thresholds': [float(t) for t in thresholds],
            'neutral_min': float(neutral_min),
            'oof_f1': float(oof_f1),
            'val_f1': float(val_f1),
            'aspect_only_f1': float(a_only_f1),
            'sent_given_aspect_acc': float(s_acc),
            'per_aspect_oof_f1': {a: pa[a]['f1'] for a in ASPECTS},
            'config': {
                'model': MODEL_NAME, 'max_len': MAX_LEN, 'batch_size': BATCH_SIZE,
                'lr': LR, 'epochs': EPOCHS, 'n_folds': N_FOLDS,
                'seeds_per_fold': SEEDS_PER_FOLD,
                'warmup_ratio': WARMUP_RATIO, 'weight_decay': WEIGHT_DECAY,
                'pos_weight_method': POS_WEIGHT_METHOD,
                'sent_weight_method': SENT_WEIGHT_METHOD,
                'seed': SEED,
            },
        }, f, indent=2)
    print("Saved thresholds_v2_2.json")

if __name__ == "__main__":
    main()