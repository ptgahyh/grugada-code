"""Step 12a: test the step-11 formulas on an independent, non-ECG multi-site dataset (UCI Heart Disease).

Data: UCI Heart Disease, processed files of Cleveland (USA), Hungarian Institute of Cardiology (Budapest) and
VA Long Beach (USA). Switzerland is excluded: only 8 of its 123 patients are negative, so a Youden threshold
cannot be estimated there. Files: analysis_v12/data/ (copy of the UCI files; md5 of Switzerland matches an
independent mirror). Outcome: angiographic disease (num > 0).

Two views of the patient, mirroring the ECG study:
  target model f   gradient boosting on clinical + resting variables (age, sex, chest-pain type, resting BP,
                   cholesterol, fasting sugar, resting ECG, max heart rate)
  anchors g        EX       ST depression on exercise (oldpeak), a fixed measurement (analogue of ST_V2)
                   EX-LR    logistic regression on the exercise test (exang, oldpeak)   (analogue of the CBM)
                   LR-same  logistic regression on f's own variables                    (analogue of LogReg9)
                   RF-same  random forest on f's own variables                          (analogue of CNNalt)
Rows with missing exercise-test values are removed (VA: 200 -> see log). Missing clinical values: source median.

Tests
  1. Threshold shift (Proposition 2): predicted Gamma/(D' - Gamma') vs observed CATT - oracle threshold.
  2. n_eff law (Proposition 3): for each setting and anchor, the labelled sample size m* at which a Youden
     threshold with true labels has the same out-of-sample loss as CATT with n unlabelled ECGs; m*/n is compared
     with the predicted n_eff/n. Subsamples are drawn from the target and evaluated on the whole target.

Run: python analysis_v12/heart_test.py analysis_v12
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common12 import bacc, gamma_shift, n_eff, youden  # noqa: E402

OUT = sys.argv[1]
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
SITES = {"cleveland": "Cleveland", "hungarian": "Hungary", "va": "VA"}
D = {}
for s in SITES:
    d = pd.read_csv(os.path.join(OUT, "data", f"processed.{s}.data"), header=None, names=COLS, na_values="?")
    n0 = len(d)
    d.loc[d.chol == 0, "chol"] = np.nan
    d = d.dropna(subset=["exang", "oldpeak"]).reset_index(drop=True)
    d["y"] = (d.num > 0).astype(int)
    D[s] = d
    print(s, n0, "->", len(d), "rows; positive", round(d.y.mean(), 3), flush=True)


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


DOM = list(SITES)
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
VSEEDS = range(10)
rows, store = [], {}
for i, (src, tgt) in enumerate(SETTINGS):
    S = pd.concat([D[s] for s in src], ignore_index=True); T = D[tgt]
    for vs in VSEEDS:
        tr, va = train_test_split(np.arange(len(S)), test_size=0.15, stratify=S.y, random_state=vs)
        med = S.iloc[tr][["age", "sex", "trestbps", "chol", "fbs", "thalach"]].median()
        Xtr, Xva, Xt = fmat(S.iloc[tr], med), fmat(S.iloc[va], med), fmat(T, med)
        ytr, yva, yt = S.y.values[tr], S.y.values[va], T.y.values
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=vs).fit(Xtr, ytr)
        fv, ft = logit(gb.predict_proba(Xva)[:, 1]), logit(gb.predict_proba(Xt)[:, 1])
        Etr, Eva, Et = (S.iloc[tr][["exang", "oldpeak"]].values, S.iloc[va][["exang", "oldpeak"]].values,
                        T[["exang", "oldpeak"]].values)
        A = {"EX": (Eva[:, 1], Et[:, 1])}
        ex = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(Etr, ytr)
        A["EX-LR"] = (ex.predict_proba(Eva)[:, 1], ex.predict_proba(Et)[:, 1])
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, ytr)
        A["LR-same"] = (lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xt)[:, 1])
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=vs).fit(Xtr, ytr)
        A["RF-same"] = (rf.predict_proba(Xva)[:, 1], rf.predict_proba(Xt)[:, 1])
        t_or = youden(yt, ft); ba_or = bacc(yt, ft, t_or); t_src = youden(yva, fv)
        for name, (gv, gt) in A.items():
            c = youden(yva, gv); pl = (gt >= c).astype(int)
            if pl.min() == pl.max():
                continue
            t_c = youden(pl, ft)
            a = pl[yt == 0].mean(); b = 1 - pl[yt == 1].mean(); q = pl.mean()
            sp, g0 = gamma_shift(ft, yt, pl, t_or)
            rows.append(dict(setting=i, source="+".join(SITES[s] for s in src), target=SITES[tgt], vseed=vs, anchor=name,
                             n=len(yt), pi=yt.mean(), alpha=a, beta=b, one_minus=1 - a - b, q=q,
                             n_eff_ratio=n_eff(len(yt), yt.mean(), a, b, q) / len(yt),
                             ba_src=bacc(yt, ft, t_src), ba_catt=bacc(yt, ft, t_c), ba_oracle=ba_or,
                             shift_obs=t_c - t_or, shift_pred=sp, gamma=g0,
                             auroc_f=float(np.mean([(ft[yt == 1][:, None] > ft[yt == 0][None, :]).mean()]))))
            if vs == 0:
                store[(i, name)] = (ft, yt, pl)
    print("setting", i, "+".join(src), "->", tgt, flush=True)
R = pd.DataFrame(rows)
R.round(5).to_csv(os.path.join(OUT, "heart_raw.csv"), index=False)
P = R.groupby(["setting", "source", "target", "anchor"], as_index=False).mean(numeric_only=True)
P.round(4).to_csv(os.path.join(OUT, "heart_points.csv"), index=False)
byA = P.groupby("anchor")[["one_minus", "n_eff_ratio", "ba_src", "ba_catt", "ba_oracle", "shift_obs", "shift_pred"]].mean()
print(byA.round(3).to_string())

# ---------------- n_eff law: matched labelled sample size
rng = np.random.default_rng(12)
DRAWS = 400
mrows = []
for (i, name), (ft, yt, pl) in store.items():
    N = len(yt); t_or = youden(yt, ft); best = bacc(yt, ft, t_or)
    pi = yt.mean(); a = pl[yt == 0].mean(); b = 1 - pl[yt == 1].mean(); q = pl.mean()
    if 1 - a - b <= 0.05:
        continue
    r_pred = n_eff(1, pi, a, b, q)

    def loss_curve(size, use_labels):
        out = []
        for _ in range(DRAWS):
            idx = rng.choice(N, size, replace=False)
            lab = yt[idx] if use_labels else pl[idx]
            if lab.min() == lab.max():
                continue
            out.append(best - bacc(yt, ft, youden(lab, ft[idx])))
        return float(np.mean(out))
    m_grid = np.unique(np.round(np.geomspace(6, N // 2, 10)).astype(int))
    lab_curve = np.array([loss_curve(m, True) for m in m_grid])
    for fr in (0.2, 0.35, 0.5):
        n_u = int(round(fr * N))
        lc = loss_curve(n_u, False)
        # interpolate log loss vs log m (labelled curve is decreasing); clip outside the grid
        lx, ly = np.log(m_grid), np.log(np.maximum(lab_curve, 1e-6))
        order = np.argsort(ly)
        m_star = float(np.exp(np.interp(np.log(max(lc, 1e-6)), ly[order], lx[order])))
        mrows.append(dict(setting=i, anchor=name, N=N, n_unlabelled=n_u, loss_catt=lc, m_star=m_star,
                          ratio_obs=m_star / n_u, ratio_pred=r_pred,
                          at_grid_edge=bool(lc >= lab_curve.max() or lc <= lab_curve.min())))
M = pd.DataFrame(mrows)
M.round(4).to_csv(os.path.join(OUT, "heart_neff.csv"), index=False)
Mv = M[~M.at_grid_edge]
indep = Mv.anchor.isin(["EX", "EX-LR"])
summary = dict(
    rows_after_cleaning={SITES[s]: len(D[s]) for s in SITES},
    by_anchor=byA.round(4).to_dict(orient="index"),
    shift_pred_vs_obs=dict(pearson=float(np.corrcoef(P.shift_pred, P.shift_obs)[0, 1]),
                           spearman=float(spearmanr(P.shift_pred, P.shift_obs)[0]),
                           slope_intercept=np.polyfit(P.shift_pred, P.shift_obs, 1).tolist(), n_points=len(P)),
    neff_independent=dict(median_obs_over_pred=float(np.median(Mv[indep].ratio_obs / Mv[indep].ratio_pred)),
                          iqr=np.percentile(Mv[indep].ratio_obs / Mv[indep].ratio_pred, [25, 75]).tolist(),
                          spearman=float(spearmanr(Mv[indep].ratio_pred, Mv[indep].ratio_obs)[0]), n=int(indep.sum())),
    neff_dependent=dict(median_obs_over_pred=float(np.median(Mv[~indep].ratio_obs / Mv[~indep].ratio_pred)),
                        iqr=np.percentile(Mv[~indep].ratio_obs / Mv[~indep].ratio_pred, [25, 75]).tolist(),
                        n=int((~indep).sum())),
    neff_points_at_grid_edge=int(M.at_grid_edge.sum()),
)
json.dump(summary, open(os.path.join(OUT, "heart_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
