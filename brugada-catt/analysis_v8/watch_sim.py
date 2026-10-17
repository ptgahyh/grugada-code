"""Simulated chest-placed smartwatch lead: train on standard-ECG archives, deploy on a watch.

Two ways of recording a precordial lead with a smartwatch (WCT = Wilson central terminal):
  RA reference (validated set-up: watch back on the chest at Vx, right index finger on the crown)
    W_x = phi(chest) - phi(RA) = Vx - (RA - WCT) = Vx + (I + II) / 3
  LA reference (watch kept on the left wrist, crown pressed on the chest at Vx)
    W_x = phi(chest) - phi(LA) = Vx - (LA - WCT) = Vx - (I - III) / 3
Orientation is chosen so that W_x has the same polarity as Vx.
The wrist-only recording is lead I.

Melo beats were released z-scored per lead (no mV), so W_x can only be built for HUCA, which has mV.
Design: train the single-lead CNN on Melo V1 or V2 (Claris, Mortara, or both), test on HUCA with
  (a) the true precordial lead Vx, (b) the simulated watch lead W_x, (c) for reference, lead I with a lead-I model.
Threshold: source Youden (SRC), CATT with a single-lead ST labeller computed on the target lead itself,
and the target oracle.

Run: python analysis_v8/watch_sim.py <huca_dir_with_metadata.csv> harmonized analysis_v8
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "concepts_v4"))
from cbm_lib4 import cnn_predict, load, pack, strat_val, train_cnn  # noqa: E402

HUCA_DIR, HD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(2)
X, Y = load(HD)
H = np.load(os.path.join(HD, "huca.npz"))

# ---- HUCA median beats in mV, same pipeline as harmonize.py but without the z-score
FS, PRE, POST, ANCHOR = 100, 29, 46, 28
HUCA_IDX = [0, 1, 2, 6, 7, 8, 9, 10, 11]
b, a = butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")


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


def zs(v):
    return (v - v.mean(-1, keepdims=True)) / (v.std(-1, keepdims=True) + 1e-8)


mv = []
for pid in H["ids"]:
    raw = np.fromfile(os.path.join(HUCA_DIR, "files", str(pid), f"{pid}.dat"), dtype="<i2")
    sig = filtfilt(b, a, raw.reshape(-1, 12)[:, HUCA_IDX].astype(float).T / 1000.0, axis=1)
    env = np.abs(np.diff(sig, axis=1)).sum(0)
    pk, _ = find_peaks(env, distance=int(0.33 * FS), height=0.35 * np.percentile(env, 99))
    pk = [p for p in pk if p - PRE >= 0 and p + POST <= sig.shape[1]]
    mv.append(align(np.median(np.stack([sig[:, p - PRE:p + POST] for p in pk]), 0)))
mv = np.stack(mv)                                                    # (356, 9, 75) mV
assert np.allclose(zs(mv), H["X"], atol=1e-3), "rebuilt HUCA beats differ from harmonized/huca.npz"

lead_I, lead_III = mv[:, 0], mv[:, 2]
la = (lead_I - lead_III) / 3.0                     # LA - WCT
ra = -(lead_I + mv[:, 1]) / 3.0                    # RA - WCT
TARGETS = {"V1": zs(mv[:, 3]), "W1-RA": zs(mv[:, 3] - ra), "W1-LA": zs(mv[:, 3] - la),
           "V2": zs(mv[:, 4]), "W2-RA": zs(mv[:, 4] - ra), "W2-LA": zs(mv[:, 4] - la), "I (wrist)": zs(lead_I)}
SRC_LEAD = {"V1": 3, "W1-RA": 3, "W1-LA": 3, "V2": 4, "W2-RA": 4, "W2-LA": 4, "I (wrist)": 0}
for nm, ref in (("LA", la), ("RA", ra)):
    print(f"|{nm}-WCT| peak relative to |V2| peak: median {np.median(np.abs(ref).max(1) / np.abs(mv[:, 4]).max(1)):.2f}")


def st(x):        # single-lead ST: 60-90 ms after QRS minus PR baseline (same windows as ST_V2)
    return x[:, 34:38].mean(1) - x[:, 14:20].mean(1)


def youden(y, s):
    f, t, th = roc_curve(y, s)
    th = np.minimum(th, s.max() + 1e-9)
    return th[np.argmax(t - f)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


yt = Y["huca"]
rows = []
for src in (["claris"], ["mortara"], ["claris", "mortara"]):
    xtr, ytr, dtr, di = pack(X, Y, src)
    tr, va = strat_val(ytr, di, 0)
    nets_by_lead = {}
    for tname, xt in TARGETS.items():
        li = SRC_LEAD[tname]
        if li not in nets_by_lead:
            a_ = xtr[:, [li]]
            nets_by_lead[li] = [train_cnn(a_[tr], ytr[tr], a_[va], ytr[va], s) for s in (0, 1, 2)]
        nets = nets_by_lead[li]
        sv = cnn_predict(nets, xtr[va][:, [li]]); s_t = cnn_predict(nets, xt[:, None, :].astype(np.float32))
        pseudo = (st(xt) >= youden(ytr[va], st(xtr[va][:, li]))).astype(int)
        rows.append(dict(train="+".join(src), target_lead=tname, auroc=roc_auc_score(yt, s_t),
                         SRC=bacc(yt, s_t, youden(ytr[va], sv)), CATT=bacc(yt, s_t, youden(pseudo, s_t)),
                         ORACLE=bacc(yt, s_t, youden(yt, s_t)), labeller_bacc=bacc(yt, pseudo, 0.5),
                         corr_with_true_lead=float(np.mean([np.corrcoef(xt[i], X["huca"][i, li])[0, 1] for i in range(len(xt))]))))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)

df = pd.DataFrame(rows)
df.round(4).to_csv(os.path.join(OUT, "watch_sim.csv"), index=False)
print(df.groupby("target_lead", sort=False)[["auroc", "SRC", "CATT", "ORACLE", "labeller_bacc", "corr_with_true_lead"]].mean().round(3).to_string())
