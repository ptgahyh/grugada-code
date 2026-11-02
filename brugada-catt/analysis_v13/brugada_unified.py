"""Step 13b: test Proposition 5 (unified loss) on the Brugada cohorts, for all 7 pseudo-labellers.

Observed: mean out-of-sample balanced-accuracy loss of CATT with n unlabelled target ECGs (n = 20/35/50% of the
target cohort, 300 subsamples without replacement, evaluated on the whole cohort) = loss_catt in
analysis_v12/brugada_neff.csv.

Predicted, from quantities of the whole target cohort (the finite population being subsampled):
  t~  CATT threshold with pseudo-labels on the whole cohort
  V~  p(t~ | y~=1)/q + p(t~ | y~=0)/(1-q)          Gaussian KDE of logit f inside each pseudo-class
  A~  |J~''(t~)|                                    from the KDE-smoothed pseudo-label Youden curve
  a   (4 V~ (1 - n/N) / (A~^2 n))^(1/3)             (1 - n/N: finite-population correction)
  NEW = mean_Z [ BA(t*) - BA(t~ + a Z) ]            BA = empirical balanced accuracy on the cohort
  OLD = [BA(t*) - BA(t~)] + quadratic n_eff term (Propositions 1 + 3), same correction
Run: python analysis_v13/brugada_unified.py analysis_v11 analysis_v12 analysis_v13
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_v12"))
from common12 import VAR_Z, bacc, density_terms, n_eff, youden  # noqa: E402

V11, V12, OUT = sys.argv[1:4]
LABS = {"CBM5": "CBM-sd15", "CBM3": "CBM", "ST_V2": "ST_V2", "LogReg9": "LogReg9", "CNN9": "CNN9",
        "ResNet": "ResNet", "CNNalt": "CNNalt"}
CONCEPT = {"CBM-sd15", "CBM", "ST_V2"}
rng = np.random.default_rng(1313)
L, H = 3.0, 0.002
s = np.arange(0, L + H, H)


def chernoff(m):
    out = np.empty(m)
    for i in range(m):
        w1 = np.concatenate([[0], np.cumsum(rng.standard_normal(len(s) - 1)) * np.sqrt(H)]) - s ** 2
        w2 = np.concatenate([[0], np.cumsum(rng.standard_normal(len(s) - 1)) * np.sqrt(H)]) - s ** 2
        out[i] = s[np.argmax(w1)] if w1.max() >= w2.max() else -s[np.argmax(w2)]
    return out


ZS = chernoff(4000)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def unified_terms(f, pl):
    """V~ and A~ at the KDE maximiser of the smoothed pseudo-label Youden curve."""
    q = pl.mean()
    k1, k0 = gaussian_kde(f[pl == 1]), gaussian_kde(f[pl == 0])
    g = np.linspace(np.percentile(f, 1), np.percentile(f, 99), 4001); dg = g[1] - g[0]
    d1, d0 = k1(g), k0(g)
    jt = np.cumsum((d0 - d1)) * dg                    # J~(t) up to a constant: F0(t) - F1(t)
    i = int(np.argmax(jt))
    A = abs(np.gradient(d0 - d1, dg)[i])              # J~'' = (p0 - p1)'
    V = d1[i] / q + d0[i] / (1 - q)
    return V, A, g[i]


obs = pd.read_csv(os.path.join(V12, "brugada_neff.csv"))
rows = []
for i in range(9):
    Z = np.load(os.path.join(V11, "scores", f"lab_{i}_0.npz"))
    yv, yt = Z["y_val"], Z["y_tgt"]; ft = logit(Z["f_tgt"]); N = len(yt)
    t_or = youden(yt, ft); best = bacc(yt, ft, t_or)
    grid_ba = lambda t: np.array([bacc(yt, ft, u) for u in np.atleast_1d(t)])  # noqa: E731
    phi, J2, _ = density_terms(ft, yt, t_or)
    for k, name in LABS.items():
        o = obs[(obs.setting == i) & (obs.anchor == name)]
        if o.empty:
            continue
        gv, gt = Z[f"{k}_val"], Z[f"{k}_tgt"]
        pl = (gt >= youden(yv, gv)).astype(int)
        a_, b_ = pl[yt == 0].mean(), 1 - pl[yt == 1].mean(); q = pl.mean()
        t_c = youden(pl, ft)
        V, A, t_kde = unified_terms(ft, pl)
        bias = best - bacc(yt, ft, t_c)
        for _, r in o.iterrows():
            n = int(r.n_unlabelled); fpc = 1 - n / N
            a = (4 * V * fpc / (A ** 2 * n)) ** (1 / 3)
            new = float(np.mean(best - grid_ba(t_c + a * ZS[:1500])))
            ne = n_eff(n, yt.mean(), a_, b_, q) / fpc
            old = bias + 0.25 * J2 * VAR_Z * (4 * phi / (J2 ** 2 * yt.mean() * (1 - yt.mean()) * ne)) ** (2 / 3)
            rows.append(dict(setting=i, anchor=name, concept=name in CONCEPT, N=N, n=n, observed=r.loss_catt,
                             pred_new=new, pred_old=old, bias_term=bias, scale_a=a))
    print("setting", i, flush=True)
D = pd.DataFrame(rows)
D.round(5).to_csv(os.path.join(OUT, "brugada_unified.csv"), index=False)
D = D[D.observed > 1e-4]


def err(d, p):
    lr = np.log(d[p] / d.observed)
    return dict(median_ratio=float(np.exp(np.median(lr))), median_abs_log_ratio=float(np.median(np.abs(lr))),
                spearman=float(spearmanr(d[p], d.observed)[0]),
                pearson_log=float(np.corrcoef(np.log(d[p]), np.log(d.observed))[0, 1]), n=len(d))


summary = {g: dict(new=err(d, "pred_new"), old=err(d, "pred_old"))
           for g, d in (("all", D), ("concept", D[D.concept]), ("blackbox", D[~D.concept]))}
json.dump(summary, open(os.path.join(OUT, "brugada_unified_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
