"""A fully reduced-lead pipeline: 1- or 2-lead CNN + a single-lead (V2) pseudo-labeller for CATT.

ST_V2 (ST level at V2, 60-90 ms after QRS, minus PR baseline; no training) is the labeller.
Its threshold is the source-validation Youden point; it labels the target ECGs; the CNN threshold
is then the Youden point against those pseudo-labels (CATT). Compared with SRC and ORACLE.

Settings 0, 1 (HUCA->Claris, HUCA->Mortara) were used for method development; 2-8 are held-out.

Run: python analysis_v8/lead_catt.py harmonized analysis_v8
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "concepts_v4"))
from cbm_lib4 import DOMAINS, cnn_predict, load, pack, strat_val, train_cnn  # noqa: E402

HD, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(2)
X, Y = load(HD)
SUBSETS = {"V2": [4], "V1+V2": [3, 4], "V1-V3": [3, 4, 5]}
SETTINGS = [([a], b) for a in DOMAINS for b in DOMAINS if a != b] + \
           [([a for a in DOMAINS if a != b], b) for b in DOMAINS]
HELD = [2, 3, 4, 5, 6, 7, 8]


def stv2(x):
    return x[:, 4, 34:38].mean(1) - x[:, 4, 14:20].mean(1)


def youden(y, s):
    f, t, th = roc_curve(y, s)
    th = np.minimum(th, s.max() + 1e-9)
    return th[np.argmax(t - f)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


rows = []
for k, (src, tgt) in enumerate(SETTINGS):
    xtr, ytr, dtr, di = pack(X, Y, src)
    tr, va = strat_val(ytr, di, 0)
    # labeller: ST_V2 at its source threshold
    t_lab = youden(ytr[va], stv2(xtr[va]))
    pseudo = (stv2(X[tgt]) >= t_lab).astype(int)
    lab_bacc = bacc(Y[tgt], stv2(X[tgt]), t_lab)
    for name, idx in SUBSETS.items():
        a, b = xtr[:, idx], X[tgt][:, idx]
        nets = [train_cnn(a[tr], ytr[tr], a[va], ytr[va], s) for s in (0, 1, 2)]
        sv, st = cnn_predict(nets, a[va]), cnn_predict(nets, b)
        r = dict(k=k, train="+".join(src), test=tgt, leads=name, auroc=roc_auc_score(Y[tgt], st),
                 SRC=bacc(Y[tgt], st, youden(ytr[va], sv)),
                 CATT=bacc(Y[tgt], st, youden(pseudo, st)),
                 ORACLE=bacc(Y[tgt], st, youden(Y[tgt], st)), labeller_bacc=lab_bacc)
        rows.append(r); print({q: (round(v, 3) if isinstance(v, float) else v) for q, v in r.items()}, flush=True)

df = pd.DataFrame(rows)
df.round(4).to_csv(os.path.join(OUT, "lead_catt_per_setting.csv"), index=False)
h = df[df.k.isin(HELD)]
summ = h.groupby("leads")[["auroc", "SRC", "CATT", "ORACLE", "labeller_bacc"]].agg(["mean", "min"]).round(3).reindex(list(SUBSETS))
print("\nHeld-out (7 settings):\n", summ.to_string())
summ.to_csv(os.path.join(OUT, "lead_catt_summary.csv"))
