"""Step 13e: direct test of the stopping rule (Corollary 5c) on the Brugada cohorts, and a second curvature
estimator (review items).

Stopping rule: from a small unlabelled pilot (m = 50 or 100 target ECGs, f + pseudo-labels only) estimate V~ and
A~, then the number of ECGs needed for SD(t_hat) <= eps:
    n_inf = 4 SD(Z)^3 V~ / (A~^2 eps^3)       (= 0.54 V~ / (A~^2 eps^3))
    n     = n_inf / (1 + n_inf / N)            (finite cohort of N, heuristic correction)
Then draw subsamples of that size and measure the SD actually achieved. Targets eps = 0.2 and 0.3 (logit scale;
the median observed SD at 20-50% of a cohort is about 0.3). Plans with n > 0.8 N are recorded as infeasible.

Curvature estimators (fixed before looking at results):
  KDE  A~ = |(p0 - p1)'| from Gaussian KDEs with Scott's bandwidth (as in brugada_unified_v2.py)
  LQ   local quadratic least-squares fit of the EMPIRICAL pseudo-label Youden curve F0_n - F1_n on the window
       t0 +- 3 * a * SD(Z) around the KDE maximiser t0; a from KDE first, then one refit with the LQ value
       (the window is the region where t_hat actually lives). Falls back to KDE if the fit is not concave.
  V~ is from the KDE densities in both cases.
Part A: population spread test with LQ (compare with KDE results of brugada_unified_v2.csv).
Part B: stopping rule, 6 pilots per configuration, 120 subsamples per plan.
Run: python analysis_v13/stopping_rule.py analysis_v11 analysis_v13
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_v12"))
from common12 import youden  # noqa: E402

V11, OUT = sys.argv[1:3]
LABS = {"CBM5": "CBM-sd15", "CBM3": "CBM", "ST_V2": "ST_V2", "LogReg9": "LogReg9", "CNN9": "CNN9",
        "ResNet": "ResNet", "CNNalt": "CNNalt"}
CONCEPT = {"CBM-sd15", "CBM", "ST_V2"}
SDZ = float(np.sqrt(0.26355))
rng = np.random.default_rng(1316)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def estimate(f, pl, n_for_window):
    """Return t0, V, A_kde, A_lq from (f, pseudo-labels)."""
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
            return t0, V, A_k, A_k
        A = 2 * abs(c2)
    return t0, V, A_k, A


def n_needed(V, A, eps, N):
    n_inf = 4 * SDZ ** 3 * V / (A ** 2 * eps ** 3)
    return n_inf / (1 + n_inf / N)


V2 = pd.read_csv(os.path.join(OUT, "brugada_unified_v2.csv"))
popA, plans = [], []
for i in range(9):
    for vs in range(3):
        Z = np.load(os.path.join(V11, "scores", f"lab_{i}_{vs}.npz"))
        yv, yt = Z["y_val"], Z["y_tgt"]; ft = logit(Z["f_tgt"]); N = len(yt)
        for k, name in LABS.items():
            gv, gt = Z[f"{k}_val"], Z[f"{k}_tgt"]
            pl = (gt >= youden(yv, gv)).astype(int)
            ref = V2[(V2.setting == i) & (V2.vseed == vs) & (V2.anchor == name)]
            if ref.empty:
                continue
            # Part A: population spread with LQ curvature
            for _, r in ref.iterrows():
                n = int(r.n); fpc = 1 - n / N
                _, V, A_k, A_l = estimate(ft, pl, n / fpc)
                popA.append(dict(setting=i, vseed=vs, anchor=name, concept=name in CONCEPT, n=n, sd_obs=r.sd_obs,
                                 sd_kde=(4 * V * fpc / (A_k ** 2 * n)) ** (1 / 3) * SDZ,
                                 sd_lq=(4 * V * fpc / (A_l ** 2 * n)) ** (1 / 3) * SDZ))
            # Part B: stopping rule from small pilots
            cache = {}

            def achieved(n):
                key = int(round(n / 10.0) * 10)
                if key not in cache:
                    th = []
                    for _ in range(120):
                        idx = rng.choice(N, key, replace=False)
                        if pl[idx].min() == pl[idx].max():
                            continue
                        th.append(youden(pl[idx], ft[idx]))
                    cache[key] = float(np.std(th))
                return cache[key]
            for m in (50, 100):
                for rep in range(6):
                    idx = rng.choice(N, m, replace=False)
                    if pl[idx].sum() < 5 or (1 - pl[idx]).sum() < 5:
                        plans.append(dict(setting=i, vseed=vs, anchor=name, pilot=m, rep=rep, status="pilot too few pseudo-positives"))
                        continue
                    for method in ("kde", "lq"):
                        for eps in (0.2, 0.3):
                            # window for LQ uses the KDE-based plan size (no target labels anywhere)
                            _, V, A_k, _ = estimate(ft[idx], pl[idx], m)
                            n0 = n_needed(V, A_k, eps, N)
                            _, V, A_k, A_l = estimate(ft[idx], pl[idx], max(n0, m))
                            A = A_k if method == "kde" else A_l
                            n = max(n_needed(V, A, eps, N), m)
                            row = dict(setting=i, vseed=vs, anchor=name, concept=name in CONCEPT, N=N, pilot=m,
                                       rep=rep, method=method, eps=eps, n_plan=n)
                            if n > 0.8 * N:
                                row["status"] = "infeasible (n > 0.8 N)"
                            else:
                                row.update(status="ok", sd_achieved=achieved(n))
                            plans.append(row)
        print("setting", i, "seed", vs, flush=True)

A_ = pd.DataFrame(popA); A_.round(5).to_csv(os.path.join(OUT, "curvature_lq_spread.csv"), index=False)
P = pd.DataFrame(plans); P.round(5).to_csv(os.path.join(OUT, "stopping_rule.csv"), index=False)


def mr(p, o):
    return float(np.exp(np.median(np.log(p / o))))


summary = dict(
    population_spread=dict(kde=dict(median_pred_over_obs=mr(A_.sd_kde, A_.sd_obs), spearman=float(spearmanr(A_.sd_kde, A_.sd_obs)[0])),
                           lq=dict(median_pred_over_obs=mr(A_.sd_lq, A_.sd_obs), spearman=float(spearmanr(A_.sd_lq, A_.sd_obs)[0]),
                                   iqr_of_ratio=np.exp(np.percentile(np.log(A_.sd_lq / A_.sd_obs), [25, 75])).tolist())),
    stopping_rule={})
ok = P[P.status == "ok"]
for (m, method, eps), d in ok.groupby(["pilot", "method", "eps"]):
    tot = P[(P.pilot == m) & (P.method == method) & (P.eps == eps)]
    summary["stopping_rule"][f"pilot{m}_{method}_eps{eps}"] = dict(
        plans=len(tot), feasible=len(d),
        median_achieved_over_target=float(np.median(d.sd_achieved / eps)),
        iqr_achieved_over_target=np.percentile(d.sd_achieved / eps, [25, 75]).tolist(),
        share_meeting_target=float((d.sd_achieved <= eps).mean()),
        share_within_20pct=float((d.sd_achieved <= 1.2 * eps).mean()),
        median_n_plan=float(d.n_plan.median()))
summary["pilot_failures"] = int((P.status == "pilot too few pseudo-positives").sum())
json.dump(summary, open(os.path.join(OUT, "stopping_rule_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
