"""Shared estimators for step 12 (testing the step-11 formulas on other data)."""
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve

VAR_Z = 0.26355


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.minimum(thr, s.max() + 1e-9)
    return thr[np.argmax(tpr - fpr)]


def bacc(y, s, t):
    p = s >= t
    return 0.5 * (p[y == 1].mean() + (~p)[y == 0].mean())


def density_terms(f, y, t):
    """phi = f0(t) ~ f1(t), J'' = |f0'(t) - f1'(t)|, D' = d/dt log f1/f0 at t, from class-wise Gaussian KDEs."""
    k0, k1 = gaussian_kde(f[y == 0]), gaussian_kde(f[y == 1])
    h = 0.05 * f.std()
    d0 = (k0(t + h) - k0(t - h))[0] / (2 * h); d1 = (k1(t + h) - k1(t - h))[0] / (2 * h)
    p0, p1 = k0(t)[0], k1(t)[0]
    return 0.5 * (p0 + p1), abs(d0 - d1), d1 / p1 - d0 / p0


def n_eff(n, pi, alpha, beta, q):
    """Step 11, Proposition 3: CATT with n unlabelled ~ Youden with n_eff labelled."""
    om = 1 - alpha - beta
    return n * pi * (1 - pi) * om ** 2 / (q * (1 - q))


def gamma_shift(f, y, pl, t, bw=0.3):
    """Step 11, Proposition 2: predicted CATT - oracle threshold shift, Gamma / (D' - Gamma')."""
    pi = y.mean(); p0, p1 = 1 - pi, pi
    a = pl[y == 0].mean(); b = 1 - pl[y == 1].mean(); om = 1 - a - b
    h = bw * np.sqrt(0.5 * (f[y == 0].var() + f[y == 1].var()))

    def gam(u):
        w = np.exp(-0.5 * ((f - u) / h) ** 2)
        al = np.sum(w * pl * (y == 0)) / np.sum(w * (y == 0))
        bl = 1 - np.sum(w * pl * (y == 1)) / np.sum(w * (y == 1))
        return ((bl - b) / p0 - (al - a) / p1) / om
    _, _, dlog = density_terms(f, y, t)
    g0 = gam(t); dg = (gam(t + 0.5 * h) - gam(t - 0.5 * h)) / h
    return g0 / (dlog - dg), g0
