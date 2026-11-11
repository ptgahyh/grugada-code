"""Bước 25a: ngưỡng làm trơn ngoài mô hình Gauss (4 kịch bản của bước 13). Xem PLAN_vi.md.

Chạy: python analysis_v25/nongauss25.py analysis_v25
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis_v24"))
from lib24 import R, bandwidth, c_const, estimators, n_needed_raw, n_needed_smooth, smooth_thr  # noqa: E402

rng = np.random.default_rng(2501)
NU = np.array([0.0, 2.5]); MID = 1.25; PI = 0.2; REPS = 1000; EST = 100


class Mix:  # như analysis_v13/nongauss_sim.py
    def __init__(self, w, mus, sds):
        self.w, self.mus, self.sds = np.array(w), np.array(mus), np.array(sds)
        g = np.linspace(-6, 10, 200001); self._g, self._c = g, self.cdf(g)

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


def population(dists, rg, rho, c, step):
    g = np.linspace(*rg, 52001); dg = g[1] - g[0]; pk = np.array([1 - PI, PI])
    fk = np.stack([d.pdf(g) for d in dists])
    u = np.stack([norm.ppf(np.clip(d.cdf(g), 1e-12, 1 - 1e-12)) for d in dists])
    ak = np.stack([norm.sf(c, NU[k] + rho * u[k], np.sqrt(1 - rho ** 2)) for k in (0, 1)])
    q = float(sum(pk[k] * norm.sf(c, NU[k], 1) for k in (0, 1)))
    d1 = (pk[:, None] * fk * ak).sum(0) / q; d0 = (pk[:, None] * fk * (1 - ak)).sum(0) / (1 - q)
    i = int(np.argmax(np.cumsum(d0 - d1)))
    tt, V, A = g[i], d1[i] / q + d0[i] / (1 - q), abs(np.gradient(d0 - d1, dg)[i])
    if step is not None:                       # khối điểm trên lưới làm tròn
        lat = np.round(g / step) * step; m1, m0 = np.zeros_like(g), np.zeros_like(g)
        for L in np.unique(lat):
            sel = lat == L; j = int(np.argmin(np.abs(g - L)))
            m1[j] += d1[sel].sum(); m0[j] += d0[sel].sum()
        d1, d0 = m1, m0
    return g, dg, d1, d0, q, tt, V, A


def exact_sd(g, dg, d1, d0, q, h, n):
    s = h / dg; f = lambda d, w: gaussian_filter1d(d, w, truncate=8, mode="constant")
    p1h, p0h, p1r, p0r = f(d1, s), f(d0, s), f(d1, s / np.sqrt(2)), f(d0, s / np.sqrt(2))
    Dh = p1h - p0h; i = int(np.argmax(np.cumsum(Dh[::-1])[::-1]))
    v1 = p1r[i] * R / h - p1h[i] ** 2; v0 = p0r[i] * R / h - p0h[i] ** 2
    return float(np.sqrt((v1 / q + v0 / (1 - q)) / n) / abs(np.gradient(Dh, dg)[i]))


def ba_fun(dists, step):
    if step is None:
        return lambda t: 0.5 * (dists[1].sf(t) + dists[0].cdf(t))
    return lambda t: (lambda e: 0.5 * (dists[1].sf(e) + dists[0].cdf(e)))(np.ceil(np.asarray(t) / step - 1e-9) * step - step / 2)


def sample(dists, step, rho, c, n):
    y = (rng.random(n) < PI).astype(int)
    z1 = rng.standard_normal(n); z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    f = np.where(y == 1, dists[1].ppf(norm.cdf(z1)), dists[0].ppf(norm.cdf(z1)))
    if step is not None:
        f = np.round(f / step) * step
    return f, (NU[y] + z2 >= c).astype(float)


def raw_thr(f, pl):
    """Ngưỡng Youden thường (quy tắc f ≥ t), xử lý đúng các giá trị trùng."""
    u, inv = np.unique(f, return_inverse=True)
    c1 = np.bincount(inv, weights=pl, minlength=len(u)); c0 = np.bincount(inv, weights=1 - pl, minlength=len(u))
    J = np.cumsum(c1[::-1])[::-1] / c1.sum() - np.cumsum(c0[::-1])[::-1] / c0.sum()
    return u[int(np.argmax(J))]


if __name__ == "__main__":
    OUT = sys.argv[1]
    cells, ests, needs = [], [], []
    for name, (dists, rg, step) in SCEN.items():
        ba = ba_fun(dists, step); best = float(np.max(ba(np.linspace(*rg, 40001))))
        for rho in (0.0, 0.6):
            for sh in (0.0, -1.25):
                g, dg, d1, d0, q, tt, V, A = population(dists, rg, rho, MID + sh, step)
                fb, plb = sample(dists, step, rho, MID + sh, 300000); cc = c_const(fb); mq = min(q, 1 - q)
                for e in (0.15, 0.30):
                    needs.append(dict(scenario=name, rho=rho, shift=sh, eps=e, n_raw=n_needed_raw(V, A, e), n_smooth=n_needed_smooth(V, A, cc, mq, e)))
                for n in (100, 300, 1000, 3000):
                    tr, ts, hs = [], [], []
                    for r in range(REPS):
                        f, pl = sample(dists, step, rho, MID + sh, n)
                        if not 3 <= pl.sum() <= n - 3:
                            continue
                        h = bandwidth(f, pl); tr.append(raw_thr(f, pl)); ts.append(smooth_thr(f, pl, h=h)); hs.append(h)
                        if r < EST:
                            e = estimators(f, pl, rng, boot=False); ests.append(dict(scenario=name, rho=rho, shift=sh, n=n, G1=e["G1_closed"], G2=e["G2_sandwich"]))
                    tr, ts = np.array(tr), np.array(ts); hm = float(np.median(hs))
                    cells.append(dict(scenario=name, rho=rho, shift=sh, n=n, used=len(tr), t_tilde=tt, h_median=hm,
                                      rmse_raw=float(np.sqrt(np.mean((tr - tt) ** 2))), rmse_smooth=float(np.sqrt(np.mean((ts - tt) ** 2))),
                                      loss_raw=float(np.mean(best - ba(tr))), loss_smooth=float(np.mean(best - ba(ts))),
                                      sd_smooth=float(ts.std()), sd_raw=float(tr.std()), pred_exact=exact_sd(g, dg, d1, d0, q, hm, n)))
                    for x in ests[-EST:]:
                        x["sd_obs"] = float(ts.std())
                print("done", name, rho, sh, flush=True)
    pd.DataFrame(cells).round(6).to_csv(os.path.join(OUT, "nongauss25_cells.csv"), index=False)
    pd.DataFrame(ests).round(6).to_csv(os.path.join(OUT, "nongauss25_est.csv"), index=False)
    pd.DataFrame(needs).round(1).to_csv(os.path.join(OUT, "nongauss25_n_needed.csv"), index=False)
