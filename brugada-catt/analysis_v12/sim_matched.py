"""Step 12c: is the 1.6x gap between observed and predicted n_eff/n a finite-sample property of the matching
procedure? Same procedure as heart_test.py / brugada_neff.py, in the Gaussian model of step 11 with an
INDEPENDENT anchor (rho = 0), where the population answer is known exactly.
Run: python analysis_v12/sim_matched.py analysis_v12"""
import json, os, sys
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common12 import n_eff, youden  # noqa: E402
OUT = sys.argv[1]
rng = np.random.default_rng(3)
MU = np.array([0.0, 1.6]); SD = np.array([1.0, 1.6]); NU = np.array([0.0, 2.5])
bacc = lambda t: 0.5 * (norm.sf(t, MU[1], SD[1]) + norm.cdf(t, MU[0], SD[0]))
TS = brentq(lambda t: norm.pdf(t, MU[0], SD[0]) - norm.pdf(t, MU[1], SD[1]), 0, 1.6); BEST = bacc(TS)
def draw(n, pi, c):
    y = (rng.random(n) < pi).astype(int); f = MU[y] + SD[y] * rng.standard_normal(n)
    g = NU[y] + rng.standard_normal(n); return y, f, (g >= c).astype(int)
def loss(n, pi, c, lab, reps=400):
    out = []
    for _ in range(reps):
        y, f, pl = draw(n, pi, c); l = y if lab else pl
        if l.min() == l.max(): continue
        out.append(BEST - bacc(youden(l, f)))
    return np.mean(out)
rows = []
for pi in (0.2, 0.4):
    for sh in (0.0, -0.75, -1.25):
        c = 1.25 + sh
        a = norm.sf(c, 0, 1); b = 1 - norm.sf(c, 2.5, 1); q = pi * (1 - b) + (1 - pi) * a
        r = n_eff(1, pi, a, b, q)
        mg = np.unique(np.round(np.geomspace(6, 400, 12)).astype(int))
        lc = np.array([loss(m, pi, c, True) for m in mg]); lx, ly = np.log(mg), np.log(lc); o = np.argsort(ly)
        for n in (70, 150, 300, 1000):
            l = loss(n, pi, c, False)
            ms = float(np.exp(np.interp(np.log(l), ly[o], lx[o])))
            rows.append(dict(pi=pi, shift=sh, n=n, ratio_pred=r, ratio_obs=ms / n, obs_over_pred=ms / n / r,
                             n_eff=r * n, edge=bool(l >= lc.max() or l <= lc.min())))
            print(rows[-1], flush=True)
json.dump(rows, open(os.path.join(OUT, "sim_matched.json"), "w"), indent=1)
