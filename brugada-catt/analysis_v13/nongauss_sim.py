"""Step 13d: Proposition 5 outside the Gaussian model it was derived in (review item).

Score f and anchor noise are linked by a Gaussian copula: latent (z1, z2) with corr rho, f = Q_k(Phi(z1)) in class
k, anchor g = nu_k + z2, pseudo-label 1[g >= c]. So any score marginals can be used and every population quantity
(t~, V~, A~, exact BA curve) is computed exactly on a grid.

Scenarios
  skewed      class 0 ~ Gamma(2, 1), class 1 ~ Gamma(5, 1)
  heavy-tail  class 0 ~ t_3(0, 1),   class 1 ~ t_3(2, 1.5)
  bimodal     class 0 ~ N(0, 1),     class 1 ~ 0.5 N(1, 0.6) + 0.5 N(3.2, 0.6)
  discrete    Gaussian model of step 11 with scores rounded to a lattice of step 0.25 (about 0.2 SD).
              The cube-root theory assumes continuous scores; this checks how it fails.
Grid: rho in {0, 0.6}, anchor shift in {0, -1.25}, n in {100, 300, 1000, 3000}, 400 replicates per cell.
Two tests: spread (SD of t_hat vs a * SD(Z)) and loss (Monte Carlo vs E_Z[BA(t*) - BA(t~ + aZ)]).
Run: python analysis_v13/nongauss_sim.py analysis_v13
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_v12"))
from common12 import youden  # noqa: E402

OUT = sys.argv[1]
ZS = np.load(os.path.join(OUT, "chernoff_samples.npy")).astype(float)
SDZ = float(np.sqrt(0.26355))
NU = np.array([0.0, 2.5]); MID = 1.25; PI = 0.2; REPS = 400
rng = np.random.default_rng(1315)


class Mix:
    def __init__(self, w, mus, sds):
        self.w, self.mus, self.sds = np.array(w), np.array(mus), np.array(sds)
        g = np.linspace(-6, 10, 200001)
        self._g, self._c = g, self.cdf(g)

    def pdf(self, x):
        return sum(w * norm.pdf(x, m, s) for w, m, s in zip(self.w, self.mus, self.sds))

    def cdf(self, x):
        return sum(w * norm.cdf(x, m, s) for w, m, s in zip(self.w, self.mus, self.sds))

    def sf(self, x):
        return 1 - self.cdf(x)

    def ppf(self, u):
        return np.interp(u, self._c, self._g)


SCEN = {
    "skewed": ([stats.gamma(2, scale=1), stats.gamma(5, scale=1)], (0.01, 16), None),
    "heavy-tail": ([stats.t(3, 0, 1), stats.t(3, 2, 1.5)], (-12, 16), None),
    "bimodal": ([stats.norm(0, 1), Mix([0.5, 0.5], [1, 3.2], [0.6, 0.6])], (-5, 6), None),
    "discrete": ([stats.norm(0, 1), stats.norm(1.6, 1.6)], (-5, 8), 0.25),
}


def population(dists, rng_, rho, c):
    g = np.linspace(*rng_, 40001); dg = g[1] - g[0]
    pk = np.array([1 - PI, PI])
    fk = np.stack([d.pdf(g) for d in dists])
    u = np.stack([norm.ppf(np.clip(d.cdf(g), 1e-12, 1 - 1e-12)) for d in dists])
    ak = np.stack([norm.sf(c, NU[k] + rho * u[k], np.sqrt(1 - rho ** 2)) for k in (0, 1)])
    q = float(sum(pk[k] * norm.sf(c, NU[k], 1) for k in (0, 1)))
    d1 = (pk[:, None] * fk * ak).sum(0) / q; d0 = (pk[:, None] * fk * (1 - ak)).sum(0) / (1 - q)
    i = int(np.argmax(np.cumsum(d0 - d1)))
    A = abs(np.gradient(d0 - d1, dg)[i]); V = d1[i] / q + d0[i] / (1 - q)
    return g[i], V, A


def ba_fun(dists, step):
    if step is None:
        return lambda t: 0.5 * (dists[1].sf(t) + dists[0].cdf(t))
    # rule round(f) >= t  <=>  f >= ceil(t/step)*step - step/2
    return lambda t: (lambda e: 0.5 * (dists[1].sf(e) + dists[0].cdf(e)))(np.ceil(np.asarray(t) / step - 1e-9) * step - step / 2)


def sample(dists, step, rho, c, n):
    y = (rng.random(n) < PI).astype(int)
    z1 = rng.standard_normal(n); z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    f = np.where(y == 1, dists[1].ppf(norm.cdf(z1)), dists[0].ppf(norm.cdf(z1)))
    if step is not None:
        f = np.round(f / step) * step
    return f, (NU[y] + z2 >= c).astype(int)


rows = []
for name, (dists, rg, step) in SCEN.items():
    ba = ba_fun(dists, step)
    gg = np.linspace(*rg, 40001); best = float(np.max(ba(gg)))
    for rho in (0.0, 0.6):
        for sh in (0.0, -1.25):
            tt, V, A = population(dists, rg, rho, MID + sh)
            for n in (100, 300, 1000, 3000):
                a = (4 * V / (A ** 2 * n)) ** (1 / 3)
                th = []
                for _ in range(REPS):
                    f, pl = sample(dists, step, rho, MID + sh, n)
                    if pl.min() == pl.max():
                        continue
                    th.append(youden(pl, f))
                th = np.array(th)
                rows.append(dict(scenario=name, rho=rho, shift=sh, n=n, sd_obs=th.std(), sd_pred=a * SDZ,
                                 loss_obs=float(np.mean(best - ba(th))),
                                 loss_pred=float(np.mean(best - ba(tt + a * ZS))),
                                 bias=best - float(ba(tt))))
            print(name, rho, sh, flush=True)
D = pd.DataFrame(rows)
D.round(6).to_csv(os.path.join(OUT, "nongauss_sim.csv"), index=False)
summ = {}
for name, d in D.groupby("scenario"):
    rs = d.sd_pred / d.sd_obs; rl = d.loss_pred / d.loss_obs
    summ[name] = dict(spread_median_pred_over_obs=float(np.median(rs)), spread_range=[float(rs.min()), float(rs.max())],
                      loss_median_pred_over_obs=float(np.median(rl)), loss_range=[float(rl.min()), float(rl.max())],
                      spread_by_n={str(k): round(float(np.median(g.sd_pred / g.sd_obs)), 3) for k, g in d.groupby("n")},
                      loss_by_n={str(k): round(float(np.median(g.loss_pred / g.loss_obs)), 3) for k, g in d.groupby("n")})
json.dump(summ, open(os.path.join(OUT, "nongauss_sim_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
