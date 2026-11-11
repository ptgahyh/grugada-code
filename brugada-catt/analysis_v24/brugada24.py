"""Bước 24, T5 + T6: Brugada (điểm số bước 8, mỏ neo CBM-sd15). Chạy: python analysis_v24/brugada24.py <analysis_v8/scores> analysis_v24"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib24 import c_const, estimators, kde_terms, n_needed_raw, n_needed_smooth, smooth_thr  # noqa: E402

SC, OUT = sys.argv[1:3]; rng = np.random.default_rng(2405)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return np.minimum(thr, s.max() + 1e-9)[np.argmax(tpr - fpr)]


rows, full = [], []
for fn in sorted(glob.glob(os.path.join(SC, "scores_*.npz"))):
    i, vs = map(int, os.path.basename(fn)[7:-4].split("_")); Z = np.load(fn)
    f = logit(Z["cnn_tgt"]); pl = (Z["cbm5_tgt"] >= youden(Z["y_val"], Z["cbm5_val"])).astype(float); N = len(f)
    o = np.argsort(f); V, A = kde_terms(f[o], pl[o]); q = pl.mean()
    full.append(dict(setting=i, vseed=vs, N=N, **{f"n_smooth_{e}": n_needed_smooth(V, A, c_const(f), min(q, 1 - q), e) for e in (0.15, 0.3)},
                     **{f"n_raw_{e}": n_needed_raw(V, A, e) for e in (0.15, 0.3)}))
    for fr in (0.2, 0.35, 0.5):
        n = int(round(fr * N)); th = []
        for _ in range(300):
            k = rng.choice(N, n, replace=False)
            if 3 <= pl[k].sum() <= n - 3:
                th.append(smooth_thr(f[k], pl[k]))
        sd_obs = float(np.std(th))
        for s in range(20):
            k = rng.choice(N, n, replace=False)
            if not 3 <= pl[k].sum() <= n - 3:
                continue
            e = estimators(f[k], pl[k], rng, boot=False)
            rows.append(dict(setting=i, vseed=vs, N=N, frac=fr, n=n, sample=s, sd_obs=sd_obs,
                             **{g: e[g] * np.sqrt(1 - fr) for g in ("G1_closed", "G2_sandwich", "G3_hybrid")}))
    print("done", i, vs, flush=True)
pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "brugada24_single.csv"), index=False)
pd.DataFrame(full).round(1).to_csv(os.path.join(OUT, "brugada24_n_needed.csv"), index=False)
