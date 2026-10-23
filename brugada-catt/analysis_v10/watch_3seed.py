"""Simulated chest-smartwatch lead with 3 validation seeds (re-run of analysis_v8/watch_sim.py, which used one).

Watch back on the chest at Vx, right finger on the crown (RA reference):  W_x = Vx + (I + II) / 3
Watch kept on the left wrist, crown on the chest (LA reference):          W_x = Vx - (I - III) / 3
Built from HUCA in mV (Melo was released z-scored). Single-lead CNN trained on Melo (Claris, Mortara,
Claris+Mortara) lead Vx, tested on the HUCA real or simulated lead. CATT uses a single-lead ST labeller
measured on the target lead itself. Scores are saved for the bootstrap.

Run: python analysis_v10/watch_3seed.py <huca_dir> harmonized analysis_v10
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VSEEDS, ScoreStore, bacc, cnn_predict, load, pack, strat_val, train_cnn, youden  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HUCA_DIR, HD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(1)
X, Y = load(HD)
H = np.load(os.path.join(HD, "huca.npz"))
FS, PRE, POST, ANCHOR = 100, 29, 46, 28
IDX = [0, 1, 2, 6, 7, 8, 9, 10, 11]
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


def zs(v):
    return (v - v.mean(-1, keepdims=True)) / (v.std(-1, keepdims=True) + 1e-8)


mv = []
for pid in H["ids"]:
    raw = np.fromfile(os.path.join(HUCA_DIR, "files", str(pid), f"{pid}.dat"), dtype="<i2")
    sig = filtfilt(*BP, raw.reshape(-1, 12)[:, IDX].astype(float).T / 1000.0, axis=1)
    env = np.abs(np.diff(sig, axis=1)).sum(0)
    pk, _ = find_peaks(env, distance=int(0.33 * FS), height=0.35 * np.percentile(env, 99))
    pk = [p for p in pk if p - PRE >= 0 and p + POST <= sig.shape[1]]
    mv.append(align(np.median(np.stack([sig[:, p - PRE:p + POST] for p in pk]), 0)))
mv = np.stack(mv)
assert np.allclose(zs(mv), H["X"], atol=1e-3)
la, ra = (mv[:, 0] - mv[:, 2]) / 3.0, -(mv[:, 0] + mv[:, 1]) / 3.0
TARGETS = {"Real V2": (4, zs(mv[:, 4])), "Watch at V2 (RA)": (4, zs(mv[:, 4] - ra)),
           "Watch at V2 (LA)": (4, zs(mv[:, 4] - la)), "Real V1": (3, zs(mv[:, 3])),
           "Watch at V1 (RA)": (3, zs(mv[:, 3] - ra)), "Watch at V1 (LA)": (3, zs(mv[:, 3] - la)),
           "Wrist (lead I)": (0, zs(mv[:, 0]))}


def st(x):
    return x[:, 34:38].mean(1) - x[:, 14:20].mean(1)


yt = Y["huca"]
store = ScoreStore(os.path.join(OUT, "watch_scores.npz"))
rows = []
for si, src in enumerate((["claris"], ["mortara"], ["claris", "mortara"])):
    xtr, ytr, dtr, di = pack(X, Y, src)
    for vs in VSEEDS:
        tr, va = strat_val(ytr, di, vs)
        nets = {}
        for name, (li, xt) in TARGETS.items():
            if li not in nets:
                a = xtr[:, [li]]
                nets[li] = [train_cnn(a[tr], ytr[tr], a[va], ytr[va], s) for s in (0, 1, 2)]
            sv = cnn_predict(nets[li], xtr[va][:, [li]]); s_t = cnn_predict(nets[li], xt[:, None, :].astype(np.float32))
            store.put(name, si, vs, sv, s_t)
            pseudo = (st(xt) >= youden(ytr[va], st(xtr[va][:, li]))).astype(int)
            rows.append(dict(train="+".join(src), vseed=vs, target=name, auroc=roc_auc_score(yt, s_t),
                             SRC=bacc(yt, s_t, youden(ytr[va], sv)), CATT=bacc(yt, s_t, youden(pseudo, s_t)),
                             ORACLE=bacc(yt, s_t, youden(yt, s_t))))
        store.save()
        print(src, vs, flush=True)
df = pd.DataFrame(rows)
df.round(4).to_csv(os.path.join(OUT, "watch_3seed_raw.csv"), index=False)
summ = df.groupby("target", sort=False)[["auroc", "SRC", "CATT", "ORACLE"]].agg(["mean", "std"]).round(3)
summ.to_csv(os.path.join(OUT, "watch_3seed_summary.csv"))
print(summ.to_string())
