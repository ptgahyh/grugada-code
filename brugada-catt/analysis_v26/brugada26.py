"""Bước 26 trên Brugada: 26a CATT có cổng; 26b so với gán nhãn k ECG đích. Xem PLAN_vi.md.

Chạy: python analysis_v26/brugada26.py <analysis_v8/scores> analysis_v26
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis_v24"))
from lib24 import estimators, kde_terms, smooth_thr  # noqa: E402

SC, OUT = sys.argv[1:3]; rng = np.random.default_rng(2601)
BROKEN, HELD, KS, ZS = [0, 1, 2, 4, 6], [2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 50, 100, 200], [1, 2, 3]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def youden(y, s):
    if y.min() == y.max():
        return None
    fpr, tpr, thr = roc_curve(y, s)
    return np.minimum(thr, s.max() + 1e-9)[np.argmax(tpr - fpr)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


def catt(f, pl):
    """(t_c, σ̂ thường, t_h, σ̂ G2) trên một lô không nhãn."""
    o = np.argsort(f); V, A = kde_terms(f[o], pl[o]); n = len(f)
    e = estimators(f, pl, rng, boot=False)
    return youden(pl, f), 0.5134 * (4 * V / (A ** 2 * n)) ** (1 / 3), e["t"], e["G2_sandwich"]


gate_rows, lab_rows = [], []
for fn in sorted(glob.glob(os.path.join(SC, "scores_*.npz"))):
    i, vs = map(int, os.path.basename(fn)[7:-4].split("_")); Z = np.load(fn)
    f, y = logit(Z["cnn_tgt"]), Z["y_tgt"].astype(int); N = len(f)
    t_src = youden(Z["y_val"], logit(Z["cnn_val"]))
    pl = (Z["cbm5_tgt"] >= youden(Z["y_val"], Z["cbm5_val"])).astype(int)
    tc, sc, th, sh = catt(f, pl)
    row = dict(setting=i, vseed=vs, ba_src=bacc(y, f, t_src), ba_oracle=bacc(y, f, youden(y, f)), ba_catt=bacc(y, f, tc),
               ba_catt_h=bacc(y, f, th), gap_raw_over_sd=abs(tc - t_src) / sc, gap_smooth_over_sd=abs(th - t_src) / sh)
    for z in ZS:
        row[f"open_raw_z{z}"] = int(abs(tc - t_src) > z * sc); row[f"open_smooth_z{z}"] = int(abs(th - t_src) > z * sh)
        row[f"ba_gate_raw_z{z}"] = bacc(y, f, tc if row[f"open_raw_z{z}"] else t_src)
        row[f"ba_gate_smooth_z{z}"] = bacc(y, f, th if row[f"open_smooth_z{z}"] else t_src)
    gate_rows.append(row)
    # 26b: tỉ lệ bệnh tự nhiên
    for k in KS:
        br, bs = [], []
        for _ in range(300):
            idx = rng.choice(N, min(k, N), replace=False); yk = y[idx]
            tr = youden(yk, f[idx]); br.append(bacc(y, f, tr if tr is not None else t_src))
            ts = smooth_thr(f[idx], yk.astype(float)) if min(yk.sum(), len(yk) - yk.sum()) >= 2 else (tr if tr is not None else t_src)
            bs.append(bacc(y, f, ts))
        lab_rows.append(dict(setting=i, vseed=vs, prev="natural", k=k, ba_lab=np.mean(br), p10_lab=np.percentile(br, 10),
                             ba_lab_h=np.mean(bs), p10_lab_h=np.percentile(bs, 10), ba_catt=row["ba_catt"], ba_catt_h=row["ba_catt_h"],
                             ba_src=row["ba_src"], exp_pos=k * y.mean()))
    # 26b: tỉ lệ bệnh 5%
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; npos = max(2, int(round(0.05 / 0.95 * len(neg))))
    acc = {k: ([], []) for k in KS}; cat = ([], []); src5 = []
    for _ in range(300):
        c = np.concatenate([neg, rng.choice(pos, npos, replace=False)]); fc, yc, plc = f[c], y[c], pl[c]
        tc5, _, th5, _ = catt(fc, plc) if 3 <= plc.sum() <= len(c) - 3 else (t_src, 0, t_src, 0)
        cat[0].append(bacc(yc, fc, tc5)); cat[1].append(bacc(yc, fc, th5)); src5.append(bacc(yc, fc, t_src))
        for k in KS:
            idx = rng.choice(len(c), min(k, len(c)), replace=False); yk = yc[idx]
            tr = youden(yk, fc[idx]); acc[k][0].append(bacc(yc, fc, tr if tr is not None else t_src))
            ts = smooth_thr(fc[idx], yk.astype(float)) if min(yk.sum(), len(yk) - yk.sum()) >= 2 else (tr if tr is not None else t_src)
            acc[k][1].append(bacc(yc, fc, ts))
    for k in KS:
        lab_rows.append(dict(setting=i, vseed=vs, prev="5pct", k=k, ba_lab=np.mean(acc[k][0]), p10_lab=np.percentile(acc[k][0], 10),
                             ba_lab_h=np.mean(acc[k][1]), p10_lab_h=np.percentile(acc[k][1], 10), ba_catt=np.mean(cat[0]),
                             ba_catt_h=np.mean(cat[1]), p10_catt_h=np.percentile(cat[1], 10), ba_src=np.mean(src5), exp_pos=k * 0.05))
    print("done", i, vs, flush=True)
G = pd.DataFrame(gate_rows); L = pd.DataFrame(lab_rows)
G.round(5).to_csv(os.path.join(OUT, "gate26_brugada.csv"), index=False); L.round(5).to_csv(os.path.join(OUT, "labelk26_brugada.csv"), index=False)
# ---------------------------------------------------------------- tổng hợp
m = G.groupby("setting").mean(); S = {"26a": {}, "26b": {}}
for kind, cc in (("raw", "ba_catt"), ("smooth", "ba_catt_h")):
    for z in ZS:
        b = m.loc[BROKEN]; keep = ((b[f"ba_gate_{kind}_z{z}"] - b.ba_src) / (b[cc] - b.ba_src)).mean()
        S["26a"][f"{kind}_z{z}"] = dict(kept_share_broken=round(float(keep), 3),
                                        opened_intact=int(G[G.setting.isin([3, 5, 7, 8])][f"open_{kind}_z{z}"].sum()),
                                        opened_broken=int(G[G.setting.isin(BROKEN)][f"open_{kind}_z{z}"].sum()),
                                        ba_heldout=round(float(m.loc[HELD][f"ba_gate_{kind}_z{z}"].mean()), 4))
S["26a"]["G1_met"] = bool(S["26a"]["raw_z2"]["kept_share_broken"] >= 0.9 and S["26a"]["smooth_z2"]["kept_share_broken"] >= 0.9)
S["26a"]["ref_heldout"] = {c: round(float(m.loc[HELD][c].mean()), 4) for c in ("ba_src", "ba_catt", "ba_catt_h", "ba_oracle")}
S["26a"]["gap_over_sd_median"] = {g: {str(k): round(float(v), 2) for k, v in m[f"gap_{g}_over_sd"].items()} for g in ("raw", "smooth")}
for prev, d in L.groupby("prev"):
    h = d[d.setting.isin(HELD)].groupby("k").mean(numeric_only=True); ref = float(h.ba_catt_h.iloc[0])
    kstar = lambda col: next((int(k) for k in h.index if h.loc[k, col] >= ref), None)
    S["26b"][prev] = dict(ba_catt=round(float(h.ba_catt.iloc[0]), 4), ba_catt_h=round(ref, 4), ba_src=round(float(h.ba_src.iloc[0]), 4),
                          k_star_raw=kstar("ba_lab"), k_star_smooth=kstar("ba_lab_h"),
                          by_k={int(k): dict(ba_lab=round(float(r.ba_lab), 4), p10_lab=round(float(r.p10_lab), 4), ba_lab_h=round(float(r.ba_lab_h), 4),
                                             p10_lab_h=round(float(r.p10_lab_h), 4), exp_pos=round(float(r.exp_pos), 1)) for k, r in h.iterrows()})
    if prev == "5pct":
        S["26b"][prev]["p10_catt_h"] = round(float(h.p10_catt_h.iloc[0]), 4)
json.dump(S, open(os.path.join(OUT, "summary_v26_brugada.json"), "w"), indent=1); print(json.dumps(S, indent=1))
