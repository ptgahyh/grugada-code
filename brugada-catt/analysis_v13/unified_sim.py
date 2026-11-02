"""Step 13: a unified loss formula for CATT, valid with or without conditional independence.

Step 11 had two separate pieces: population bias (Proposition 1, any anchor) and sampling error (Proposition 3,
independent anchor only). Step 12 showed two weaknesses: (i) for dependent anchors the n_eff law over-states
the sampling error (correlated errors reduce variance); (ii) at small n the quadratic loss approximation
over-states the loss. Proposition 5 (this step) fixes both:

  t_hat - t~  ~  a * Z,    a = (4 V~ / (A~^2 n))^(1/3),   Z ~ Chernoff
  V~ = p(t~ | y~=1)/q + p(t~ | y~=0)/(1-q)      (density of f in each pseudo-class at the CATT threshold t~)
  A~ = |d^2/dt^2 J~(t~)|                          (curvature of the pseudo-label Youden curve at t~)
  expected balanced-accuracy loss = E_Z[ BA(t*) - BA(t~ + a Z) ]   (exact BA curve, no quadratic expansion)

Under independence t~ = t*, V~ = phi/(q(1-q)), A~ = k|J''|, which gives back n_eff (Proposition 3).

Checks in the Gaussian model of step 11 (pi = 0.2): rho in {0, 0.3, 0.6, 0.9}, anchor shift in {0, -0.75, -1.25,
-1.75}, n in {100, 300, 1000, 3000}; Monte Carlo 500 replicates per cell. Compared predictors:
  OLD  = BA loss at t~ (Prop. 1) + quadratic n_eff sampling term (Prop. 3)
  NEW  = Proposition 5
Run: python analysis_v13/unified_sim.py analysis_v13
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_v12"))
from common12 import youden  # noqa: E402

OUT = sys.argv[1]
PI = 0.2
MU = np.array([0.0, 1.6]); SD = np.array([1.0, 1.6]); NU = np.array([0.0, 2.5]); MID = 1.25
VAR_Z = 0.26355
rng = np.random.default_rng(13)
GRID = np.linspace(-4, 7, 22001); DT = GRID[1] - GRID[0]


def ba(t):
    return 0.5 * (norm.sf(t, MU[1], SD[1]) + norm.cdf(t, MU[0], SD[0]))


TS = GRID[np.argmax(ba(GRID))]; BEST = ba(TS)


def chernoff_samples(m=20000, L=3.0, h=0.002):
    """argmax of two-sided Brownian motion minus s^2 on [-L, L] (discretised)."""
    s = np.arange(0, L + h, h)
    out = np.empty(m)
    for i in range(m):
        w1 = np.concatenate([[0], np.cumsum(rng.standard_normal(len(s) - 1)) * np.sqrt(h)]) - s ** 2
        w2 = np.concatenate([[0], np.cumsum(rng.standard_normal(len(s) - 1)) * np.sqrt(h)]) - s ** 2
        out[i] = s[np.argmax(w1)] if w1.max() >= w2.max() else -s[np.argmax(w2)]
    return out


ZS = chernoff_samples(6000)
print("Chernoff sample variance", round(ZS.var(), 4), "(theory 0.2636)", flush=True)


def population(rho, c):
    f = np.stack([norm.pdf(GRID, MU[k], SD[k]) for k in (0, 1)])
    a = np.stack([norm.sf(c, NU[k] + rho * (GRID - MU[k]) / SD[k], np.sqrt(1 - rho ** 2)) for k in (0, 1)])
    pk = np.array([1 - PI, PI])
    q = float(sum(pk[k] * norm.sf(c, NU[k], 1) for k in (0, 1)))
    dens1 = (pk[:, None] * f * a).sum(0) / q            # density of f in pseudo-positives
    dens0 = (pk[:, None] * f * (1 - a)).sum(0) / (1 - q)
    surv1 = np.cumsum(dens1[::-1])[::-1] * DT; surv0 = np.cumsum(dens0[::-1])[::-1] * DT
    jt = surv1 - surv0
    i = int(np.argmax(jt)); tt = GRID[i]
    w = 200
    cur = abs(np.polyfit(GRID[i - w:i + w] - tt, jt[i - w:i + w], 2)[0] * 2)
    V = dens1[i] / q + dens0[i] / (1 - q)
    alpha = norm.sf(c, NU[0], 1); beta = 1 - norm.sf(c, NU[1], 1)
    return dict(t_tilde=tt, A=cur, V=V, q=q, alpha=alpha, beta=beta)


phi = norm.pdf(TS, MU[0], SD[0])
J2 = abs(np.gradient(np.gradient(2 * ba(GRID) - 1, DT), DT)[np.argmin(abs(GRID - TS))])


def predict(pop, n):
    a = (4 * pop["V"] / (pop["A"] ** 2 * n)) ** (1 / 3)
    new = float(np.mean(BEST - ba(pop["t_tilde"] + a * ZS)))
    om = 1 - pop["alpha"] - pop["beta"]; q = pop["q"]
    n_eff = n * PI * (1 - PI) * om ** 2 / (q * (1 - q))
    old = float(BEST - ba(pop["t_tilde"])) + 0.25 * J2 * VAR_Z * (4 * phi / (J2 ** 2 * PI * (1 - PI) * n_eff)) ** (2 / 3)
    return new, old


def mc(rho, c, n, reps=500):
    out = []
    for _ in range(reps):
        y = (rng.random(n) < PI).astype(int)
        z1 = rng.standard_normal(n); z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        f = MU[y] + SD[y] * z1; pl = (NU[y] + z2 >= c).astype(int)
        if pl.min() == pl.max():
            continue
        out.append(BEST - ba(youden(pl, f)))
    return float(np.mean(out))


rows = []
for rho in (0.0, 0.3, 0.6, 0.9):
    for sh in (0.0, -0.75, -1.25, -1.75):
        pop = population(rho, MID + sh)
        for n in (100, 300, 1000, 3000):
            new, old = predict(pop, n)
            obs = mc(rho, MID + sh, n)
            rows.append(dict(rho=rho, shift=sh, n=n, observed=obs, pred_new=new, pred_old=old,
                             t_tilde=pop["t_tilde"], bias_loss=BEST - ba(pop["t_tilde"])))
        print("rho", rho, "shift", sh, flush=True)
D = pd.DataFrame(rows)
D.round(5).to_csv(os.path.join(OUT, "unified_sim.csv"), index=False)


def err(p):
    return dict(median_abs_log_ratio=float(np.median(np.abs(np.log(D[p] / D.observed)))),
                max_abs_log_ratio=float(np.max(np.abs(np.log(D[p] / D.observed)))),
                median_ratio=float(np.median(D[p] / D.observed)),
                r2_log=float(1 - np.var(np.log(D.observed) - np.log(D[p])) / np.var(np.log(D.observed))))


summary = dict(new=err("pred_new"), old=err("pred_old"),
               new_by_rho={str(r): float(np.median(g.pred_new / g.observed)) for r, g in D.groupby("rho")},
               old_by_rho={str(r): float(np.median(g.pred_old / g.observed)) for r, g in D.groupby("rho")},
               new_by_n={str(r): float(np.median(g.pred_new / g.observed)) for r, g in D.groupby("n")},
               old_by_n={str(r): float(np.median(g.pred_old / g.observed)) for r, g in D.groupby("n")},
               chernoff_var=float(ZS.var()))
json.dump(summary, open(os.path.join(OUT, "unified_sim_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
