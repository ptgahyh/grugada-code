"""Bước 18b-Brugada (mô tả): các ước lượng không nhãn của SD(t̂) trên toàn bộ cohort đích, mỏ neo CBM-sd15.

Dùng điểm số bước 8 (không nằm trong kho vì có nhãn từng bệnh nhân). Không dùng nhãn đích.
Chạy: python analysis_v18/est_brugada.py <thư mục analysis_v8/scores> analysis_v18
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common18 import estimators  # noqa: E402

SC, OUT = sys.argv[1:3]
rng = np.random.default_rng(1803)


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
    o = np.argsort(f); e = estimators(f[o], pl[o], rng)
    rows.append(dict(setting=i, vseed=vs, N=len(f), q=float(pl.mean()), ties=int(len(f) - len(np.unique(f))), **e))
D = pd.DataFrame(rows); D.round(5).to_csv(os.path.join(OUT, "est_brugada.csv"), index=False)
E = [c for c in D.columns if c.startswith("E")]
R = D[E].div(D.E1_half, axis=0)
print(pd.DataFrame({"median_ratio_to_F1": R.median(), "min": R.min(), "max": R.max()}).round(3))
print(D.groupby("setting")[["N", "q"] + E].mean().round(3))
