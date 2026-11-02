"""Step 13f: Proposition 5 (spread of the CATT threshold, Theorem 5a; label-free plug-in, Corollary 5c) on the
UCI Heart Disease data of step 12 (Cleveland, Hungary, VA; 9 source -> target settings; 4 anchors).

Same models and anchors as analysis_v12/heart_test.py, validation seeds 0-2. For each setting/anchor and
n = 20/35/50% of the target: SD of the CATT threshold over 300 subsamples vs a * SD(Z), with V~ from KDE and
A~ from KDE or local quadratic (LQ, see stopping_rule.py); plus the label-free plug-in from one subsample (median of
20). Caveats: targets are small (144-303), so n is 29-152; gradient-boosting scores have many ties.
Run: python analysis_v13/heart_unified.py analysis_v12 analysis_v13
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_v12"))
from common12 import youden  # noqa: E402

V12, OUT = sys.argv[1:3]
SDZ = float(np.sqrt(0.26355))
rng = np.random.default_rng(1317)
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
SITES = {"cleveland": "Cleveland", "hungarian": "Hungary", "va": "VA"}
D = {}
for s in SITES:
    d = pd.read_csv(os.path.join(V12, "data", f"processed.{s}.data"), header=None, names=COLS, na_values="?")
    d.loc[d.chol == 0, "chol"] = np.nan
    d = d.dropna(subset=["exang", "oldpeak"]).reset_index(drop=True)
    d["y"] = (d.num > 0).astype(int)
    D[s] = d


def fmat(d, med):
    x = d[["age", "sex", "trestbps", "chol", "fbs", "thalach"]].copy().fillna(med)
    for v in (1, 2, 3, 4):
        x[f"cp{v}"] = (d.cp == v).astype(float)
    for v in (0, 1, 2):
        x[f"ecg{v}"] = (d.restecg.fillna(0) == v).astype(float)
    return x.values.astype(float)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def estimate(f, pl, n_for_window):
    q = pl.mean()
    g = np.linspace(np.percentile(f, 1), np.percentile(f, 99), 1201); dg = g[1] - g[0]
    d1, d0 = gaussian_kde(f[pl == 1])(g), gaussian_kde(f[pl == 0])(g)
    i = int(np.argmax(np.cumsum(d0 - d1))); i = min(max(i, 2), len(g) - 3)
    t0 = g[i]; V = d1[i] / q + d0[i] / (1 - q); A_k = abs(np.gradient(d0 - d1, dg)[i])
    f1, f0 = np.sort(f[pl == 1]), np.sort(f[pl == 0])
    A = A_k
    for _ in range(2):
        a = (4 * V / (A ** 2 * n_for_window)) ** (1 / 3)
        w = 3 * a * SDZ
        tt = np.linspace(t0 - w, t0 + w, 61)
        J = np.searchsorted(f0, tt, side="left") / len(f0) - np.searchsorted(f1, tt, side="left") / len(f1)
        c2 = np.polyfit(tt - t0, J, 2)[0]
        if c2 >= 0:
            return V, A_k, A_k
        A = 2 * abs(c2)
    return V, A_k, A


DOM = list(SITES)
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
rows = []
for i, (src, tgt) in enumerate(SETTINGS):
    S = pd.concat([D[s] for s in src], ignore_index=True); T = D[tgt]
    for vs in range(3):
        tr, va = train_test_split(np.arange(len(S)), test_size=0.15, stratify=S.y, random_state=vs)
        med = S.iloc[tr][["age", "sex", "trestbps", "chol", "fbs", "thalach"]].median()
        Xtr, Xva, Xt = fmat(S.iloc[tr], med), fmat(S.iloc[va], med), fmat(T, med)
        ytr, yva = S.y.values[tr], S.y.values[va]
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=vs).fit(Xtr, ytr)
        ft = logit(gb.predict_proba(Xt)[:, 1]); N = len(ft)
        Etr, Eva, Et = (S.iloc[tr][["exang", "oldpeak"]].values, S.iloc[va][["exang", "oldpeak"]].values,
                        T[["exang", "oldpeak"]].values)
        A = {"EX": (Eva[:, 1], Et[:, 1])}
        ex = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(Etr, ytr)
        A["EX-LR"] = (ex.predict_proba(Eva)[:, 1], ex.predict_proba(Et)[:, 1])
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, ytr)
        A["LR-same"] = (lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xt)[:, 1])
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=vs).fit(Xtr, ytr)
        A["RF-same"] = (rf.predict_proba(Xva)[:, 1], rf.predict_proba(Xt)[:, 1])
        for name, (gv, gt) in A.items():
            pl = (gt >= youden(yva, gv)).astype(int)
            if pl.sum() < 10 or (1 - pl).sum() < 10:
                continue
            for fr in (0.2, 0.35, 0.5):
                n = int(round(fr * N)); fpc = 1 - n / N
                th = []
                for _ in range(300):
                    idx = rng.choice(N, n, replace=False)
                    if pl[idx].min() == pl[idx].max():
                        continue
                    th.append(youden(pl[idx], ft[idx]))
                V, A_k, A_l = estimate(ft, pl, n / fpc)
                plug = []
                for _ in range(20):
                    idx = rng.choice(N, n, replace=False)
                    if pl[idx].sum() < 5 or (1 - pl[idx]).sum() < 5:
                        continue
                    Vp, Akp, _ = estimate(ft[idx], pl[idx], n)
                    plug.append((4 * Vp * fpc / (Akp ** 2 * n)) ** (1 / 3) * SDZ)
                rows.append(dict(setting=i, vseed=vs, anchor=name, N=N, n=n, sd_obs=float(np.std(th)),
                                 sd_kde=(4 * V * fpc / (A_k ** 2 * n)) ** (1 / 3) * SDZ,
                                 sd_lq=(4 * V * fpc / (A_l ** 2 * n)) ** (1 / 3) * SDZ,
                                 sd_plugin=float(np.median(plug)) if plug else np.nan,
                                 distinct_score_share=len(np.unique(ft)) / N))
    print("setting", i, flush=True)
R = pd.DataFrame(rows)
R.round(5).to_csv(os.path.join(OUT, "heart_unified.csv"), index=False)


def mr(p, o):
    ok = np.isfinite(p) & (o > 0)
    return float(np.exp(np.median(np.log(p[ok] / o[ok]))))


def sp(p, o):
    ok = np.isfinite(p)
    return float(spearmanr(p[ok], o[ok])[0])


summ = {m: dict(median_pred_over_obs=mr(R[m], R.sd_obs), spearman=sp(R[m], R.sd_obs),
                iqr_of_ratio=np.exp(np.nanpercentile(np.log(R[m] / R.sd_obs), [25, 75])).tolist())
        for m in ("sd_kde", "sd_lq", "sd_plugin")}
summ["n_points"] = len(R)
summ["by_anchor_kde"] = R.groupby("anchor").apply(lambda d: mr(d.sd_kde, d.sd_obs)).round(3).to_dict()
summ["median_distinct_score_share"] = float(R.distinct_score_share.median())
json.dump(summ, open(os.path.join(OUT, "heart_unified_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
