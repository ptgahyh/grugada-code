"""Bước 25b: ngưỡng làm trơn trên Brugada với 5 mỏ neo (điểm số bước 8). Xem PLAN_vi.md.

Chạy: python analysis_v25/brugada25.py <analysis_v8/scores> analysis_v25
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis_v24"))
from lib24 import c_const, estimators, kde_terms, n_needed_raw, n_needed_smooth, smooth_thr  # noqa: E402
sys.path.insert(0, HERE)
from nongauss25 import raw_thr  # noqa: E402  (chỉ dùng hàm; phần chạy được chặn bởi __main__ bên dưới)

SC, OUT = sys.argv[1:3]; rng = np.random.default_rng(2502)
ANCHORS = ["cbm5", "cbm3", "stv2", "dann", "coral"]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return np.minimum(thr, s.max() + 1e-9)[np.argmax(tpr - fpr)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


ok = lambda p: 3 <= p.sum() <= len(p) - 3
spread, acc, single, need, skipped = [], [], [], [], []
for fn in sorted(glob.glob(os.path.join(SC, "scores_*.npz"))):
    i, vs = map(int, os.path.basename(fn)[7:-4].split("_")); Z = np.load(fn)
    f = logit(Z["cnn_tgt"]); y = Z["y_tgt"].astype(int); N = len(f)
    for an in ANCHORS:
        pl = (Z[f"{an}_tgt"] >= youden(Z["y_val"], Z[f"{an}_val"])).astype(float)
        if not ok(pl):
            skipped.append(dict(setting=i, vseed=vs, anchor=an)); continue
        o = np.argsort(f); V, A = kde_terms(f[o], pl[o]); q = pl.mean()
        need.append(dict(setting=i, vseed=vs, anchor=an, **{f"n_raw_{e}": n_needed_raw(V, A, e) for e in (0.15, 0.3)},
                         **{f"n_smooth_{e}": n_needed_smooth(V, A, c_const(f), min(q, 1 - q), e) for e in (0.15, 0.3)}))
        row = dict(setting=i, vseed=vs, anchor=an, ba_full_raw=bacc(y, f, raw_thr(f, pl)), ba_full_smooth=bacc(y, f, smooth_thr(f, pl)))
        for n in (50, 100):
            br, bs = [], []
            for _ in range(200):
                k = rng.choice(N, n, replace=False)
                if ok(pl[k]):
                    br.append(bacc(y, f, raw_thr(f[k], pl[k]))); bs.append(bacc(y, f, smooth_thr(f[k], pl[k])))
            row[f"ba_n{n}_raw"], row[f"ba_n{n}_smooth"] = float(np.mean(br)), float(np.mean(bs))
        acc.append(row)
        for fr in (0.2, 0.35, 0.5):
            n = int(round(fr * N)); tr, ts = [], []
            for _ in range(300):
                k = rng.choice(N, n, replace=False)
                if ok(pl[k]):
                    tr.append(raw_thr(f[k], pl[k])); ts.append(smooth_thr(f[k], pl[k]))
            spread.append(dict(setting=i, vseed=vs, anchor=an, frac=fr, sd_raw=float(np.std(tr)), sd_smooth=float(np.std(ts))))
            for s in range(20):
                k = rng.choice(N, n, replace=False)
                if ok(pl[k]):
                    e = estimators(f[k], pl[k], rng, boot=False)
                    single.append(dict(setting=i, vseed=vs, anchor=an, frac=fr, sd_obs=float(np.std(ts)), G1=e["G1_closed"] * np.sqrt(1 - fr),
                                       G2=e["G2_sandwich"] * np.sqrt(1 - fr)))
    print("done", i, vs, flush=True)
for nm, d in (("spread", spread), ("acc", acc), ("single", single), ("n_needed", need), ("skipped", skipped)):
    pd.DataFrame(d).round(6).to_csv(os.path.join(OUT, f"brugada25_{nm}.csv"), index=False)
