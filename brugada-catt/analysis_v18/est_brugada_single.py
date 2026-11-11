"""Bước 18c: độ chính xác TỪNG MẪU ĐƠN của các ước lượng không nhãn trên Brugada (xem bổ sung 18c trong PLAN_vi.md).

Chạy: python analysis_v18/est_brugada_single.py <thư mục analysis_v8/scores> analysis_v18
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common18 import curves, estimators, subsample_masks  # noqa: E402

SC, OUT = sys.argv[1:3]
rng = np.random.default_rng(1804)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return np.minimum(thr, s.max() + 1e-9)[np.argmax(tpr - fpr)]


rows = []
for fn in sorted(glob.glob(os.path.join(SC, "scores_*.npz"))):
    i, vs = map(int, os.path.basename(fn)[7:-4].split("_")); Z = np.load(fn)
    f = logit(Z["cnn_tgt"]); pl = (Z["cbm5_tgt"] >= youden(Z["y_val"], Z["cbm5_val"])).astype(float)
    o = np.argsort(f); f, pl = f[o], pl[o]; N = len(f)
    for fr in (0.2, 0.35, 0.5):
        n = int(round(fr * N)); J, ok = curves(subsample_masks(rng, 300, N, n), pl)
        sd_obs = float(np.std(f[np.argmax(J, 1)][ok]))
        for s in range(20):
            idx = np.sort(rng.choice(N, n, replace=False)); e = estimators(f[idx], pl[idx], rng)
            if e is not None:
                rows.append(dict(setting=i, vseed=vs, N=N, frac=fr, n=n, sample=s, sd_obs=sd_obs, **e))
    print("done", i, vs, flush=True)
pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "est_brugada_single.csv"), index=False)
