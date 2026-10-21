"""Within-HUCA 10-fold cross-validation, to compare with studies that report HUCA-only results
(e.g. An et al., CinC 2026: 10-fold CV, AUROC 0.87, sensitivity 0.67, specificity 0.92, F1 0.66, accuracy 0.87).

CNN on V1-V3 and on 9 leads, 3-seed ensemble per fold. The threshold is the Youden point on an inner
validation split of the training folds (never on the test fold). Note: we use 356 records (duplicates and
flat-line records removed); with duplicates, a patient can appear in both training and test folds.

Run: python analysis_v8/huca_cv10.py harmonized analysis_v8
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "concepts_v4"))
from cbm_lib4 import cnn_predict, load, train_cnn  # noqa: E402

HD, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(1)
X, Y = load(HD)
x, y = X["huca"], Y["huca"]
INPUTS = {"CNN V1-V3": [3, 4, 5], "CNN 9 leads": list(range(9))}


def youden(yy, s):
    f, t, th = roc_curve(yy, s)
    th = np.minimum(th, s.max() + 1e-9)
    return th[np.argmax(t - f)]


rows = []
for rep in (0, 1, 2):
    skf = StratifiedKFold(10, shuffle=True, random_state=rep)
    for name, idx in INPUTS.items():
        s_all, p_all = np.zeros(len(y)), np.zeros(len(y), bool)
        for f, (trv, te) in enumerate(skf.split(x, y)):
            tr, va = next(StratifiedKFold(5, shuffle=True, random_state=100 * rep + f).split(trv, y[trv]))
            tr, va = trv[tr], trv[va]
            a = x[:, idx]
            nets = [train_cnn(a[tr], y[tr], a[va], y[va], s) for s in (0, 1, 2)]
            t = youden(y[va], cnn_predict(nets, a[va]))
            s_all[te] = cnn_predict(nets, a[te]); p_all[te] = s_all[te] >= t
        sens, spec = p_all[y == 1].mean(), (~p_all[y == 0]).mean()
        rows.append(dict(rep=rep, model=name, auroc=roc_auc_score(y, s_all), sens=sens, spec=spec,
                         bacc=(sens + spec) / 2, f1=f1_score(y, p_all), acc=(p_all == (y == 1)).mean()))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)

df = pd.DataFrame(rows)
df.round(4).to_csv(os.path.join(OUT, "huca_cv10_raw.csv"), index=False)
summ = df.groupby("model")[["auroc", "sens", "spec", "bacc", "f1", "acc"]].agg(["mean", "std"]).round(3)
print(summ.to_string())
summ.to_csv(os.path.join(OUT, "huca_cv10_summary.csv"))
