"""Device-only shift with repeated splits (re-run of analysis_v8/device_sim.py, which used one split).

Same HUCA patients and labels; only the acquisition chain is simulated (see analysis_v8/device_sim.py):
LP25 (25 Hz low-pass), HP0.67c / HP1c (causal 1st-order high-pass, monitor-style), quant+noise
(20 uV quantization + 20 uV RMS noise), combined.
A. Within HUCA: 5-fold CV repeated 3 times (different fold seeds), threshold from original ECGs.
B. Melo -> HUCA: 3 source choices x 3 validation seeds.
Shift metric: median logit of HUCA negatives minus median logit of source-validation negatives, in units
of the source class separation. The noise variant is re-drawn for every repetition.

Run: python analysis_v10/device_3seed.py <huca_dir> harmonized analysis_v10
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, find_peaks, lfilter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VSEEDS, bacc, cnn_predict, load, pack, strat_val, train_cnn, youden  # noqa: E402

HUCA_DIR, HD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(1)
X, Y = load(HD)
H = np.load(os.path.join(HD, "huca.npz"))
FS, PRE, POST, ANCHOR = 100, 29, 46, 28
IDX = [0, 1, 2, 6, 7, 8, 9, 10, 11]
V13 = [3, 4, 5]
BP = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")


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


def lp25(s, rng):
    b, a = butter(4, 25 / (FS / 2)); return filtfilt(b, a, s, axis=1)


def hpc(fc):
    def f(s, rng):
        b, a = butter(1, fc / (FS / 2), btype="high"); return lfilter(b, a, s, axis=1)
    return f


def qn(s, rng):
    return np.round((s + rng.normal(0, 0.020, s.shape)) / 0.020) * 0.020


VAR = {"orig": lambda s, r: s, "LP25": lp25, "HP0.67c": hpc(0.67), "HP1c": hpc(1.0), "quant+noise": qn,
       "combined": lambda s, r: qn(hpc(0.67)(lp25(s, r), r), r)}


def beat(raw, dev, rng):
    sig = filtfilt(*BP, dev(raw, rng), axis=1)
    env = np.abs(np.diff(sig, axis=1)).sum(0)
    pk, _ = find_peaks(env, distance=int(0.33 * FS), height=0.35 * np.percentile(env, 99))
    pk = [p for p in pk if p - PRE >= 0 and p + POST <= sig.shape[1]]
    return zs(align(np.median(np.stack([sig[:, p - PRE:p + POST] for p in pk]), 0)))


raws = [np.fromfile(os.path.join(HUCA_DIR, "files", str(p), f"{p}.dat"), dtype="<i2").reshape(-1, 12)[:, IDX]
        .astype(float).T / 1000.0 for p in H["ids"]]


def variants(rep):
    rng = np.random.default_rng(rep)
    return {k: np.stack([beat(r, f, rng) for r in raws]).astype(np.float32) for k, f in VAR.items()}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def shift(sv, yv, st, yt):
    sep = np.median(logit(sv[yv == 1])) - np.median(logit(sv[yv == 0]))
    return (np.median(logit(st[yt == 0])) - np.median(logit(sv[yv == 0]))) / sep


yh = Y["huca"]
rows = []
for rep in VSEEDS:
    XV = variants(rep)
    assert np.allclose(XV["orig"], H["X"], atol=1e-3)
    # A. within HUCA
    skf = StratifiedKFold(5, shuffle=True, random_state=rep)
    sc = {k: np.zeros(len(yh)) for k in VAR}; thr = np.zeros(len(yh)); svs, yvs = [], []
    for f, (trv, te) in enumerate(skf.split(XV["orig"], yh)):
        tr, va = next(StratifiedKFold(4, shuffle=True, random_state=10 * rep + f).split(trv, yh[trv]))
        tr, va = trv[tr], trv[va]
        xo = XV["orig"][:, V13]
        nets = [train_cnn(xo[tr], yh[tr], xo[va], yh[va], s) for s in (0, 1, 2)]
        sv = cnn_predict(nets, xo[va]); thr[te] = youden(yh[va], sv); svs.append(sv); yvs.append(yh[va])
        for k in VAR:
            sc[k][te] = cnn_predict(nets, XV[k][te][:, V13])
    sv_all, yv_all = np.concatenate(svs), np.concatenate(yvs)
    for k in VAR:
        s = sc[k]
        rows.append(dict(part="A within HUCA", rep=rep, source="HUCA", variant=k, auroc=roc_auc_score(yh, s),
                         SRC=np.mean([bacc(yh[te], s[te], thr[te][0]) for _, te in skf.split(XV["orig"], yh)]),
                         neg_shift=shift(sv_all, yv_all, s, yh)))
    # B. Melo -> HUCA
    for src in (["claris"], ["mortara"], ["claris", "mortara"]):
        xtr, ytr, dtr, di = pack(X, Y, src)
        tr, va = strat_val(ytr, di, rep)
        a = xtr[:, V13]
        nets = [train_cnn(a[tr], ytr[tr], a[va], ytr[va], s) for s in (0, 1, 2)]
        sv = cnn_predict(nets, a[va]); t = youden(ytr[va], sv)
        for k in VAR:
            s = cnn_predict(nets, XV[k][:, V13])
            rows.append(dict(part="B Melo->HUCA", rep=rep, source="+".join(src), variant=k, auroc=roc_auc_score(yh, s),
                             SRC=bacc(yh, s, t), neg_shift=shift(sv, ytr[va], s, yh)))
    print("rep", rep, "done", flush=True)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "device_3seed_raw.csv"), index=False)

df = pd.DataFrame(rows)
summ = df.groupby(["part", "variant"], sort=False)[["auroc", "SRC", "neg_shift"]].agg(["mean", "std"]).round(3)
summ.to_csv(os.path.join(OUT, "device_3seed_summary.csv"))
print(summ.to_string())
