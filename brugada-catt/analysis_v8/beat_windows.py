"""Where in the beat is the cross-site Brugada signal? (the "hint" question)

The median beat has 75 samples at 100 Hz with the QRS energy peak at sample 28
(sample 0 = -280 ms). The CNN-V1V3 is retrained with everything outside one time window
set to 0 (the z-scored mean), same protocol as the main study (9 settings, 3-seed ensemble).

Windows (ms relative to the QRS peak):
  full        -280..+460   whole beat
  PR          -280..-70    P wave and PR segment
  QRS          -60..+50
  earlyST      +50..+130   J point to ~J+80 ms
  ST           +50..+210
  T           +210..+460
  QRS+eST      -60..+130   200 ms around the J point
  noT         -280..+210   whole beat without the T wave

Run: python analysis_v8/beat_windows.py harmonized analysis_v8
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "concepts_v4"))
from cbm_lib4 import DOMAINS, cnn_predict, load, pack, strat_val, train_cnn  # noqa: E402

HD, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(2)
X, Y = load(HD)
LEADS = [3, 4, 5]
WIN = {"full": (0, 75), "PR": (0, 22), "QRS": (22, 34), "earlyST": (33, 42), "ST": (33, 50), "T": (49, 75),
       "QRS+eST": (22, 42), "noT": (0, 50)}
SETTINGS = [([a], b) for a in DOMAINS for b in DOMAINS if a != b] + \
           [([a for a in DOMAINS if a != b], b) for b in DOMAINS]


def mask(x, w):
    m = np.zeros_like(x); m[:, :, w[0]:w[1]] = x[:, :, w[0]:w[1]]
    return m


def youden(y, s):
    f, t, th = roc_curve(y, s)
    th = np.minimum(th, s.max() + 1e-9)
    return th[np.argmax(t - f)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


rows, t0 = [], time.time()
for k, (src, tgt) in enumerate(SETTINGS):
    xtr, ytr, dtr, di = pack(X, Y, src)
    tr, va = strat_val(ytr, di, 0)
    for name, w in WIN.items():
        a, b = mask(xtr[:, LEADS], w), mask(X[tgt][:, LEADS], w)
        nets = [train_cnn(a[tr], ytr[tr], a[va], ytr[va], s) for s in (0, 1, 2)]
        sv, st = cnn_predict(nets, a[va]), cnn_predict(nets, b)
        r = dict(k=k, train="+".join(src), test=tgt, window=name, auroc=roc_auc_score(Y[tgt], st),
                 SRC=bacc(Y[tgt], st, youden(ytr[va], sv)), ORACLE=bacc(Y[tgt], st, youden(Y[tgt], st)),
                 val_auroc=roc_auc_score(ytr[va], sv))
        rows.append(r); print({q: (round(v, 3) if isinstance(v, float) else v) for q, v in r.items()},
                              f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "beat_windows_per_setting.csv"), index=False)

df = pd.DataFrame(rows)
df["gap"] = df.val_auroc - df.auroc          # source-validation minus target AUROC: how much is lost by moving site
summ = df.groupby("window")[["val_auroc", "auroc", "gap", "SRC", "ORACLE"]].agg(["mean", "min"]).round(3).reindex(list(WIN))
print(summ.to_string())
summ.to_csv(os.path.join(OUT, "beat_windows_summary.csv"))
print("done", f"{time.time() - t0:.0f}s")
