"""Device-only shift: same HUCA patients, simulated acquisition chains of other ECG machines.

Question (advisor): is the threshold collapse just a machine problem (sampling, filters)?
Here the patients and labels are held fixed and only the acquisition chain changes. Each variant is
applied to the raw 100 Hz HUCA signal; the study's own pipeline (0.5-40 Hz zero-phase band-pass,
median beat, alignment, per-lead z-score) is then run unchanged.

Variants
  orig        as recorded
  LP25        25 Hz low-pass (zero-phase), "muscle filter" setting
  HP0.67c     0.67 Hz causal high-pass (1st order, non-linear phase), monitor-style; distorts the ST segment
  HP1c        1 Hz causal high-pass, aggressive monitor setting
  quant+noise 20 uV quantization + 20 uV RMS white noise (lower-resolution front end)
  combined    LP25 + HP0.67c + quant+noise

A. Within HUCA (5-fold CV, CNN-V1V3, 3 seeds): train and set the Youden threshold on original ECGs,
   test the held-out fold under every variant. Threshold shift is isolated from population and labels.
B. Melo -> HUCA (Claris, Mortara, Claris+Mortara as source): HUCA target under every variant.
Shift metric: median logit of HUCA negatives minus the median logit of source-validation negatives,
in units of the source class separation (median positive logit - median negative logit).

Run: python analysis_v8/device_sim.py <huca_dir_with_metadata.csv> harmonized analysis_v8
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, find_peaks, lfilter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "concepts_v4"))
from cbm_lib4 import cnn_predict, load, pack, strat_val, train_cnn  # noqa: E402

HUCA_DIR, HD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(2)
X, Y = load(HD)
H = np.load(os.path.join(HD, "huca.npz"))
FS, PRE, POST, ANCHOR = 100, 29, 46, 28
HUCA_IDX = [0, 1, 2, 6, 7, 8, 9, 10, 11]
V13 = [3, 4, 5]
BP = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
rng = np.random.default_rng(0)


def align(beat):
    shift = ANCHOR - int(np.argmax(np.abs(np.diff(beat, axis=1)).sum(0)))
    if shift == 0:
        return beat
    out = np.empty_like(beat)
    if shift > 0:
        out[:, shift:] = beat[:, :-shift]; out[:, :shift] = beat[:, :1]
    else:
        out[:, :shift] = beat[:, -shift:]; out[:, shift:] = beat[:, -1:]
    return out


def zs(b):
    return (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-8)


def lp25(s):
    b, a = butter(4, 25 / (FS / 2)); return filtfilt(b, a, s, axis=1)


def hp_causal(fc):
    def f(s):
        b, a = butter(1, fc / (FS / 2), btype="high"); return lfilter(b, a, s, axis=1)
    return f


def quant_noise(s):
    q = 0.020
    return np.round((s + rng.normal(0, 0.020, s.shape)) / q) * q


VARIANTS = {"orig": lambda s: s, "LP25": lp25, "HP0.67c": hp_causal(0.67), "HP1c": hp_causal(1.0),
            "quant+noise": quant_noise, "combined": lambda s: quant_noise(hp_causal(0.67)(lp25(s)))}


def beat(raw_mv, dev):
    sig = filtfilt(*BP, dev(raw_mv), axis=1)
    env = np.abs(np.diff(sig, axis=1)).sum(0)
    pk, _ = find_peaks(env, distance=int(0.33 * FS), height=0.35 * np.percentile(env, 99))
    pk = [p for p in pk if p - PRE >= 0 and p + POST <= sig.shape[1]]
    return zs(align(np.median(np.stack([sig[:, p - PRE:p + POST] for p in pk]), 0)))


raws = [np.fromfile(os.path.join(HUCA_DIR, "files", str(pid), f"{pid}.dat"), dtype="<i2")
        .reshape(-1, 12)[:, HUCA_IDX].astype(float).T / 1000.0 for pid in H["ids"]]
XV = {k: np.stack([beat(r, f) for r in raws]).astype(np.float32) for k, f in VARIANTS.items()}
assert np.allclose(XV["orig"], H["X"], atol=1e-3)
for k in VARIANTS:
    c = np.mean([np.corrcoef(XV[k][i, 4], XV["orig"][i, 4])[0, 1] for i in range(len(raws))])
    print(f"{k:12s} mean correlation of V2 beat with original: {c:.3f}")
yh = Y["huca"]


def youden(y, s):
    f, t, th = roc_curve(y, s)
    th = np.minimum(th, s.max() + 1e-9)
    return th[np.argmax(t - f)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def shift(sv, yv, st, yt):
    sep = np.median(logit(sv[yv == 1])) - np.median(logit(sv[yv == 0]))
    return (np.median(logit(st[yt == 0])) - np.median(logit(sv[yv == 0]))) / sep


rows = []
# ---- A. within HUCA, device swap only
skf = StratifiedKFold(5, shuffle=True, random_state=0)
scores = {k: np.zeros(len(yh)) for k in VARIANTS}; thr = np.zeros(len(yh)); sval = []
for f, (tr_all, te) in enumerate(skf.split(XV["orig"], yh)):
    tr, va = next(StratifiedKFold(4, shuffle=True, random_state=f).split(tr_all, yh[tr_all]))
    tr, va = tr_all[tr], tr_all[va]
    xo = XV["orig"][:, V13]
    nets = [train_cnn(xo[tr], yh[tr], xo[va], yh[va], s) for s in (0, 1, 2)]
    sv = cnn_predict(nets, xo[va]); t = youden(yh[va], sv); thr[te] = t; sval.append((sv, yh[va]))
    for k in VARIANTS:
        scores[k][te] = cnn_predict(nets, XV[k][te][:, V13])
sv_all = np.concatenate([a for a, _ in sval]); yv_all = np.concatenate([b for _, b in sval])
for k in VARIANTS:
    s = scores[k]
    rows.append(dict(part="A within-HUCA", source="HUCA (5-fold)", variant=k, auroc=roc_auc_score(yh, s),
                     SRC=np.mean([bacc(yh[te], s[te], thr[te][0]) for _, te in skf.split(XV["orig"], yh)]),
                     ORACLE=bacc(yh, s, youden(yh, s)), neg_shift=shift(sv_all, yv_all, s, yh)))
    print(rows[-1], flush=True)

# ---- B. Melo -> HUCA under each device variant
for src in (["claris"], ["mortara"], ["claris", "mortara"]):
    xtr, ytr, dtr, di = pack(X, Y, src)
    tr, va = strat_val(ytr, di, 0)
    a = xtr[:, V13]
    nets = [train_cnn(a[tr], ytr[tr], a[va], ytr[va], s) for s in (0, 1, 2)]
    sv = cnn_predict(nets, a[va]); t = youden(ytr[va], sv)
    for k in VARIANTS:
        s = cnn_predict(nets, XV[k][:, V13])
        rows.append(dict(part="B Melo->HUCA", source="+".join(src), variant=k, auroc=roc_auc_score(yh, s),
                         SRC=bacc(yh, s, t), ORACLE=bacc(yh, s, youden(yh, s)), neg_shift=shift(sv, ytr[va], s, yh)))
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.round(4).to_csv(os.path.join(OUT, "device_sim.csv"), index=False)
print(df.groupby(["part", "variant"], sort=False)[["auroc", "SRC", "ORACLE", "neg_shift"]].mean().round(3).to_string())
