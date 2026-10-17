"""Cross-site performance of the CNN with 1-, 2- and 3-lead inputs.

Question: can Brugada be detected across hospitals from one or two precordial leads
(the setting of a chest-worn patch or a smartwatch held at the precordium)?

Same CNN, augmentation, validation split and 9 cross-site settings as the main study;
3-seed ensemble per setting. Reported per subset and setting:
  AUROC on the target, balanced accuracy at the source Youden threshold (SRC),
  and balanced accuracy at the target-oracle Youden threshold (ORACLE).

Run: python analysis_v8/lead_subsets.py harmonized analysis_v8
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "concepts_v4"))
from cbm_lib4 import DOMAINS, cnn_predict, load, pack, strat_val, train_cnn  # noqa: E402

HD, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(2)
X, Y = load(HD)
LEADS = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
SUBSETS = {"V1": [3], "V2": [4], "V3": [5], "V1+V2": [3, 4], "V2+V3": [4, 5], "V1+V3": [3, 5],
           "V1-V3": [3, 4, 5], "I": [0], "II": [1], "9-lead": list(range(9))}
SETTINGS = [([a], b) for a in DOMAINS for b in DOMAINS if a != b] + \
           [([a for a in DOMAINS if a != b], b) for b in DOMAINS]


def youden_t(y, s):
    f, t, th = roc_curve(y, s)
    return th[np.argmax(t - f)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p[y == 0]).mean())


rows, t0 = [], time.time()
for src, tgt in SETTINGS:
    xtr, ytr, dtr, di = pack(X, Y, src)
    tr, va = strat_val(ytr, di, 0)
    for name, idx in SUBSETS.items():
        a, b = xtr[:, idx], X[tgt][:, idx]
        nets = [train_cnn(a[tr], ytr[tr], a[va], ytr[va], s) for s in (0, 1, 2)]
        s_src_va = cnn_predict(nets, a[va]); s_tgt = cnn_predict(nets, b)
        t_src = youden_t(ytr[va], s_src_va)
        r = dict(train="+".join(src), test=tgt, leads=name,
                 auroc=round(roc_auc_score(Y[tgt], s_tgt), 4),
                 bacc_src=round(bacc(Y[tgt], s_tgt, t_src), 4),
                 bacc_oracle=round(bacc(Y[tgt], s_tgt, youden_t(Y[tgt], s_tgt)), 4))
        rows.append(r); print(r, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(rows, open(os.path.join(OUT, "lead_subsets.json"), "w"), indent=1)

import pandas as pd  # noqa: E402
df = pd.DataFrame(rows)
summ = df.groupby("leads")[["auroc", "bacc_src", "bacc_oracle"]].agg(["mean", "min"]).round(3)
summ = summ.reindex(list(SUBSETS))
print(summ.to_string())
summ.to_csv(os.path.join(OUT, "lead_subsets_summary.csv"))
print("done", f"{time.time() - t0:.0f}s")
