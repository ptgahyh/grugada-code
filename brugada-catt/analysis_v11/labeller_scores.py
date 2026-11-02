"""Step 11: per-patient scores of every CATT pseudo-labeller (needed to test the bias theory on real data).

Step 6 (concepts_v6/compare_v6.py) trained the black-box labellers but stored only summary numbers. This script
re-trains them with exactly the same code and seeds and saves source-validation and target scores for:
  target model CNN-V1V3 (from concepts_v5, the paper's scores), concept labellers CBM5 (= CBM-sd15), CBM3 (= CBM),
  ST_V2 (fixed feature), black-box labellers CNN9, ResNet, LogReg9, CNNalt (CNN-V1V3 with seeds 3-5).
Reproducibility: alpha/beta of every labeller are compared with the step-6 values.

Run: python analysis_v11/labeller_scores.py harmonized concepts_v5 analysis_v11 <chunk> <n_chunks>
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v6"))
import cbm_lib4 as L  # noqa: E402
from arch_v4_lib import train_resnet  # noqa: E402

torch.set_num_threads(1)
HD, V5, OUT, CH, NCH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
X, Y = L.load(HD)
DOM = L.DOMAINS
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
V5ROWS = {(r["setting"], r["vseed"]): r for f in ["v5_raw_0.jsonl", "v5_raw_1.jsonl"]
          for r in map(json.loads, open(os.path.join(V5, f)))}


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.minimum(thr, s.max() + 1e-9)
    return thr[np.argmax(tpr - fpr)]


def st_v2(x):
    return x[:, 4, 34:38].mean(1) - x[:, 4, 14:20].mean(1)


def predict(nets, x):
    with torch.no_grad():
        return np.mean([torch.sigmoid(n(torch.tensor(x))).numpy() for n in nets], 0)


JOBS = [(i, vs) for i in range(len(SETTINGS)) for vs in (0, 1, 2)][CH::NCH]
t0 = time.time()
for i, vs in JOBS:
    fo = os.path.join(OUT, "scores", f"lab_{i}_{vs}.npz")
    if os.path.exists(fo):
        continue
    src, tgt = SETTINGS[i]
    xtr, ytr, dtr, di = L.pack(X, Y, src)
    tr, va = L.strat_val(ytr, di, vs)
    yv, xv, xt, yt = ytr[va], xtr[va], X[tgt], Y[tgt]
    v5 = V5ROWS[(i, vs)]
    assert v5["y_val"] == yv.tolist()
    S = {"f": (np.array(v5["models"]["CNN-V1V3"]["val"]), np.array(v5["models"]["CNN-V1V3"]["tgt"])),
         "CBM5": (np.array(v5["models"]["CBM5-Sign"]["val"]), np.array(v5["models"]["CBM5-Sign"]["tgt"])),
         "CBM3": (np.array(v5["models"]["CBM3-Sign"]["val"]), np.array(v5["models"]["CBM3-Sign"]["tgt"])),
         "ST_V2": (st_v2(xv), st_v2(xt))}
    n9 = [L.train_cnn(xtr[tr], ytr[tr], xv, yv, s) for s in (0, 1, 2)]
    S["CNN9"] = (predict(n9, xv), predict(n9, xt))
    rn = [train_resnet(xtr[tr][:, 3:6], ytr[tr], xv[:, 3:6], yv, s) for s in (0, 1, 2)]
    S["ResNet"] = (predict(rn, xv[:, 3:6]), predict(rn, xt[:, 3:6]))
    lr9 = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1], cv=5, scoring="roc_auc",
                                                               class_weight="balanced", max_iter=3000))
    lr9.fit(xtr[tr].reshape(len(tr), -1), ytr[tr])
    S["LogReg9"] = (lr9.predict_proba(xv.reshape(len(xv), -1))[:, 1], lr9.predict_proba(xt.reshape(len(xt), -1))[:, 1])
    na = [L.train_cnn(xtr[tr][:, 3:6], ytr[tr], xv[:, 3:6], yv, s) for s in (3, 4, 5)]
    S["CNNalt"] = (predict(na, xv[:, 3:6]), predict(na, xt[:, 3:6]))
    out = dict(y_val=yv, y_tgt=yt)
    for k, (a, b) in S.items():
        out[f"{k}_val"] = np.asarray(a, float); out[f"{k}_tgt"] = np.asarray(b, float)
    os.makedirs(os.path.dirname(fo), exist_ok=True)
    np.savez_compressed(fo, **out)
    ab = {k: (round(float((S[k][1] >= youden(yv, S[k][0]))[yt == 0].mean()), 3),
              round(float((S[k][1] < youden(yv, S[k][0]))[yt == 1].mean()), 3)) for k in S if k != "f"}
    print(i, "+".join(src), "->", tgt, "vseed", vs, ab, f"{time.time() - t0:.0f}s", flush=True)
print("done", f"{time.time() - t0:.0f}s")
