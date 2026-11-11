"""Bước 20, S2: ngưỡng làm trơn so với ngưỡng thường trên Brugada (điểm số bước 8, mỏ neo CBM-sd15).

Chạy: python analysis_v20/smooth_brugada.py <analysis_v8/scores> analysis_v20
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smooth_lib import raw_thr, smooth_thr  # noqa: E402

SC, OUT = sys.argv[1:3]
rng = np.random.default_rng(2002)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return np.minimum(thr, s.max() + 1e-9)[np.argmax(tpr - fpr)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


METH = {"raw": raw_thr, "s1.0": lambda f, p: smooth_thr(f, p, 1.0), "s0.5": lambda f, p: smooth_thr(f, p, 0.5),
        "s2.0": lambda f, p: smooth_thr(f, p, 2.0)}
sp, acc = [], []
for fn in sorted(glob.glob(os.path.join(SC, "scores_*.npz"))):
    i, vs = map(int, os.path.basename(fn)[7:-4].split("_")); Z = np.load(fn)
    f = logit(Z["cnn_tgt"]); y = Z["y_tgt"].astype(int)
    pl = (Z["cbm5_tgt"] >= youden(Z["y_val"], Z["cbm5_val"])).astype(float)
    o = np.argsort(f); f, y, pl = f[o], y[o], pl[o]; N = len(f)
    t_src = youden(Z["y_val"], logit(Z["cnn_val"])); t_or = youden(y, f)
    row = dict(setting=i, vseed=vs, N=N, ba_src=bacc(y, f, t_src), ba_oracle=bacc(y, f, t_or))
    for k, fun in METH.items():
        row[f"ba_full_{k}"] = bacc(y, f, fun(f, pl))
    for n in (50, 100):
        B = {k: [] for k in METH}
        for _ in range(200):
            idx = np.sort(rng.choice(N, n, replace=False))
            if pl[idx].sum() < 3 or n - pl[idx].sum() < 3:
                continue
            for k, fun in METH.items():
                B[k].append(bacc(y, f, fun(f[idx], pl[idx])))
        for k in METH:
            row[f"ba_n{n}_{k}"] = float(np.mean(B[k])); row[f"p10_n{n}_{k}"] = float(np.percentile(B[k], 10))
        row[f"used_n{n}"] = len(B["raw"])
    acc.append(row)
    for fr in (0.2, 0.35, 0.5):
        n = int(round(fr * N)); T = {k: [] for k in METH}
        for _ in range(300):
            idx = np.sort(rng.choice(N, n, replace=False))
            if pl[idx].sum() < 3 or n - pl[idx].sum() < 3:
                continue
            for k, fun in METH.items():
                T[k].append(fun(f[idx], pl[idx]))
        sp.append(dict(setting=i, vseed=vs, N=N, frac=fr, n=n, **{f"sd_{k}": float(np.std(v)) for k, v in T.items()}))
    print("done", i, vs, flush=True)
pd.DataFrame(acc).round(5).to_csv(os.path.join(OUT, "smooth_brugada_acc.csv"), index=False)
pd.DataFrame(sp).round(5).to_csv(os.path.join(OUT, "smooth_brugada_spread.csv"), index=False)
