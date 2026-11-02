"""Step 11: checks of the CATT bias theory in the Gaussian model (Section "Theory" of README_vi.md).

Model (same as analysis_v8/sim_catt.py and the manuscript, Section 2.9):
  y ~ Bernoulli(pi); target score f | y=k ~ N(mu_k, sd_k^2) with mu = (0, 1.6), sd = (1, 1.6);
  anchor score g | y=k ~ N(nu_k, 1) with nu = (0, 2.5); corr(f, g | y) = rho; pseudo-label = 1[g >= c],
  c = 1.25 + shift (1.25 = midpoint of the anchor's classes).

Checks
  A. Population threshold: root of the stationarity condition (Proposition 1) and its first-order form
     (Proposition 2) against Monte Carlo with n = 400,000.
  B. Population bias and balanced-accuracy loss over rho x shift (dependence alone, miscalibration alone, both).
  C. Sampling law for an independent anchor (Proposition 3): loss ~ C * n_eff^(-2/3),
     n_eff = n * pi(1-pi)(1-alpha-beta)^2 / (q(1-q)); fitted exponents in n and in (1-alpha-beta).
  D. Anchor-threshold invariance (Proposition 4): spread of the CATT threshold when c is swept, by rho.

Run: python analysis_v11/theory_sim.py analysis_v11
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

OUT = sys.argv[1]
PI = 0.2
MU = np.array([0.0, 1.6]); SD = np.array([1.0, 1.6])
NU = np.array([0.0, 2.5]); MID = 1.25
VAR_Z = 0.26355          # variance of Chernoff's distribution (argmax of two-sided Brownian motion minus s^2)


def f_pdf(t, k):
    return norm.pdf(t, MU[k], SD[k])


def bacc(t):
    return 0.5 * (norm.sf(t, MU[1], SD[1]) + norm.cdf(t, MU[0], SD[0]))


T_STAR = brentq(lambda t: f_pdf(t, 0) - f_pdf(t, 1), 0.0, 1.6)        # true Youden threshold (density crossing)
BEST = bacc(T_STAR)
J2 = abs((-(T_STAR - MU[0]) / SD[0] ** 2) * f_pdf(T_STAR, 0) - (-(T_STAR - MU[1]) / SD[1] ** 2) * f_pdf(T_STAR, 1))
PHI = f_pdf(T_STAR, 0)


def local_rate(t, k, rho, c):
    """a_k(t) = P(g >= c | f = t, y = k)."""
    m = NU[k] + rho * (t - MU[k]) / SD[k]
    return norm.sf(c, m, np.sqrt(1 - rho ** 2)) if rho < 1 else (m >= c).astype(float)


def rates(c):
    alpha = norm.sf(c, NU[0], 1); one_minus_beta = norm.sf(c, NU[1], 1)
    q = PI * one_minus_beta + (1 - PI) * alpha
    return alpha, 1 - one_minus_beta, q


def stationarity(t, rho, c):
    """Proposition 1: sum_k pi_k f_k(t) (a_k(t) - q) = 0 at the population CATT threshold."""
    _, _, q = rates(c)
    return PI * f_pdf(t, 1) * (local_rate(t, 1, rho, c) - q) + (1 - PI) * f_pdf(t, 0) * (local_rate(t, 0, rho, c) - q)


def pop_threshold(rho, c):
    lo, hi = -2.0, 4.0
    ts = np.linspace(lo, hi, 2401); v = stationarity(ts, rho, c)
    # dJ~/dt = -S(t)/(q(1-q)), so the maximiser is where S changes sign from - to +; take the root nearest to t*
    roots = [brentq(stationarity, ts[i], ts[i + 1], args=(rho, c)) for i in range(len(ts) - 1) if v[i] < 0 <= v[i + 1]]
    return min(roots, key=lambda r: abs(r - T_STAR)) if roots else np.nan


def gamma_fn(t, rho, c):
    alpha, beta, q = rates(c)
    a0 = local_rate(t, 0, rho, c); a1 = local_rate(t, 1, rho, c)
    return np.log((q - a0) / (q - alpha)) - np.log((a1 - q) / (1 - beta - q))


def first_order(rho, c):
    """Proposition 2: t - t* ~= Gamma(t*) / (D'(t*) - Gamma'(t*)), with
    Gamma(t) = log[(q - a0(t)) / (q - alpha)] - log[(a1(t) - q) / (1 - beta - q)]  (zero under independence)
    D(t) = log f1(t)/f0(t) (the target model's own log likelihood ratio, zero at t*)."""
    g0 = gamma_fn(T_STAR, rho, c)
    h = 1e-4
    dg = (gamma_fn(T_STAR + h, rho, c) - gamma_fn(T_STAR - h, rho, c)) / (2 * h)
    dlog = (-(T_STAR - MU[1]) / SD[1] ** 2) - (-(T_STAR - MU[0]) / SD[0] ** 2)
    return T_STAR + g0 / (dlog - dg), float(g0), float(local_rate(T_STAR, 0, rho, c)), float(local_rate(T_STAR, 1, rho, c))


def youden_thr(y, s):
    o = np.argsort(-s); ss = s[o]; yy = y[o]
    tp = np.cumsum(yy) / max(yy.sum(), 1); fp = np.cumsum(1 - yy) / max((1 - yy).sum(), 1)
    j = np.argmax(tp - fp)
    return 0.5 * (ss[j] + ss[min(j + 1, len(ss) - 1)])


def sample(rng, n, rho, c):
    y = (rng.random(n) < PI).astype(int)
    z1 = rng.standard_normal(n); z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    f = MU[y] + SD[y] * z1
    g = NU[y] + z2
    return y, f, (g >= c).astype(int)


rng = np.random.default_rng(11)
SHIFTS = [0.75, 0.0, -0.75, -1.25, -1.75, -2.25]
RHOS = [0.0, 0.3, 0.6, 0.9]

# ---------------- A + B: population threshold, first-order formula, Monte Carlo
rowsA = []
for rho in RHOS:
    for sh in SHIFTS:
        c = MID + sh
        alpha, beta, q = rates(c)
        t_pop = pop_threshold(rho, c)
        t_lin, gamma, a0, a1 = first_order(rho, c)
        mc = [youden_thr(*(lambda y, f, p: (p, f))(*sample(rng, 400_000, rho, c))) for _ in range(3)]
        rowsA.append(dict(rho=rho, shift=sh, alpha=alpha, beta=beta, one_minus=1 - alpha - beta, q=q,
                          local_alpha=a0, local_one_minus_beta=a1, gamma=gamma,
                          t_star=T_STAR, t_pop=t_pop, t_first_order=t_lin, t_mc=float(np.mean(mc)), t_mc_sd=float(np.std(mc)),
                          bias=t_pop - T_STAR, loss_pop=BEST - bacc(t_pop), loss_first_order=BEST - bacc(t_lin)))
A = pd.DataFrame(rowsA)
A.round(5).to_csv(os.path.join(OUT, "theory_population.csv"), index=False)
print(A[["rho", "shift", "one_minus", "t_pop", "t_first_order", "t_mc", "bias", "loss_pop"]].round(4).to_string(index=False))

# ---------------- C: sampling law, independent anchor
NS = [100, 200, 400, 800, 1600, 3200, 6400]
REPS = 600
rowsC = []
CACHE = os.path.join(OUT, "theory_sampling.csv")
for sh in ([] if os.path.exists(CACHE) else [0.0, -0.75, -1.25, -1.75, -2.25]):
    c = MID + sh
    alpha, beta, q = rates(c)
    om = 1 - alpha - beta
    for n in NS:
        loss = []
        for _ in range(REPS):
            y, f, pl = sample(rng, n, 0.0, c)
            if pl.min() == pl.max():
                continue
            loss.append(BEST - bacc(youden_thr(pl, f)))
        loss_or = []
        for _ in range(REPS // 2):
            y, f, pl = sample(rng, n, 0.0, c)
            loss_or.append(BEST - bacc(youden_thr(y, f)))
        n_eff = n * PI * (1 - PI) * om ** 2 / (q * (1 - q))
        pred = 0.25 * J2 * VAR_Z * (4 * PHI / (J2 ** 2 * PI * (1 - PI) * n_eff)) ** (2 / 3)
        rowsC.append(dict(shift=sh, n=n, one_minus=om, q=q, n_eff=n_eff, loss_catt=float(np.mean(loss)),
                          loss_oracle_youden=float(np.mean(loss_or)), loss_pred=pred))
    print("C shift", sh, "done", flush=True)
C = pd.DataFrame(rowsC) if rowsC else pd.read_csv(CACHE)
C["loss_pred"] = 0.25 * J2 * VAR_Z * (4 * PHI / (J2 ** 2 * PI * (1 - PI) * C.n_eff)) ** (2 / 3)
C["loss_pred_oracle"] = 0.25 * J2 * VAR_Z * (4 * PHI / (J2 ** 2 * PI * (1 - PI) * C.n)) ** (2 / 3)
C.round(6).to_csv(os.path.join(OUT, "theory_sampling.csv"), index=False)
X = np.column_stack([np.ones(len(C)), np.log(C.n), np.log(C.one_minus), np.log(C.q * (1 - C.q))])
coef = np.linalg.lstsq(X, np.log(C.loss_catt), rcond=None)[0]
big = C[C.n >= 800]
Xb = np.column_stack([np.ones(len(big)), np.log(big.n), np.log(big.one_minus), np.log(big.q * (1 - big.q))])
coef_b = np.linalg.lstsq(Xb, np.log(big.loss_catt), rcond=None)[0]

# ---------------- D: anchor-threshold invariance (label-free diagnostic)
rowsD = []
SWEEP = np.linspace(-1.0, 1.0, 9)
for rho in RHOS:
    for sh in [0.0, -1.25]:
        spread_pop = np.ptp([pop_threshold(rho, MID + sh + d) for d in SWEEP])
        rowsD.append(dict(rho=rho, shift=sh, spread_population=spread_pop))
D = pd.DataFrame(rowsD)


def sweep_batch(n, rho, sh, reps=200):
    out = []
    for _ in range(reps):
        y = (rng.random(n) < PI).astype(int)
        z1 = rng.standard_normal(n); z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        f = MU[y] + SD[y] * z1; g = NU[y] + z2
        thr = []
        for d in SWEEP:
            pl = (g >= MID + sh + d).astype(int)
            if pl.min() == pl.max():
                continue
            thr.append(np.mean(f >= youden_thr(pl, f)))       # threshold expressed as fraction flagged (label-free)
        out.append(np.std(thr))
    return float(np.mean(out)), float(np.percentile(out, 90))


D["spread_n1000_mean"], D["spread_n1000_p90"] = zip(*[sweep_batch(1000, r, s) for r, s in zip(D.rho, D["shift"])])
D.round(4).to_csv(os.path.join(OUT, "theory_invariance.csv"), index=False)
print(D.round(4).to_string(index=False))

summary = dict(t_star=T_STAR, oracle_bacc=BEST, J2=J2, phi=PHI,
               max_abs_pop_vs_mc=float(np.max(np.abs(A.t_pop - A.t_mc))),
               max_abs_firstorder_vs_pop=float(np.max(np.abs(A.t_first_order - A.t_pop))),
               rho0_max_abs_bias=float(np.max(np.abs(A[A.rho == 0].bias))),
               sampling_fit_all={"log_n": coef[1], "log_one_minus": coef[2], "log_q1q": coef[3]},
               sampling_fit_n_ge_800={"log_n": coef_b[1], "log_one_minus": coef_b[2], "log_q1q": coef_b[3]},
               theory_exponents={"log_n": -2 / 3, "log_one_minus": -4 / 3, "log_q1q": 2 / 3},
               pred_over_obs_median=float(np.median(C.loss_pred / C.loss_catt)),
               pred_over_obs_median_n_ge_800=float(np.median((C.loss_pred / C.loss_catt)[C.n >= 800])),
               oracle_pred_over_obs_median=float(np.median(C.loss_pred_oracle / C.loss_oracle_youden)))
json.dump(summary, open(os.path.join(OUT, "theory_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
