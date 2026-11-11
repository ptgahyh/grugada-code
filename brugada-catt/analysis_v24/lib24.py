"""Bước 24: công thức SD của ngưỡng làm trơn, các ước lượng không nhãn G1–G4, quy tắc cỡ mẫu. Xem PLAN_vi.md."""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis_v20")); sys.path.insert(0, os.path.join(HERE, "..", "analysis_v18"))
from common18 import MU, NU, PI, SD, kde_terms  # noqa: E402,F401
from smooth_lib import bandwidth, smooth_boot_sd, smooth_thr  # noqa: E402,F401

R = 1 / (2 * np.sqrt(np.pi))


def c_const(f):
    q75, q25 = np.percentile(f, [75, 25])
    return 0.9 * min(f.std(), (q75 - q25) / 1.34)


# ---------------------------------------------------------------- quần thể (mô hình Gauss của bài)
GRID = np.linspace(-8, 11, 38001); DG = GRID[1] - GRID[0]


def pop_densities(rho, c):
    """Mật độ của f trong lớp nhãn giả dương/âm và q (như analysis_v13/unified_sim.population)."""
    f = np.stack([norm.pdf(GRID, MU[k], SD[k]) for k in (0, 1)])
    a = np.stack([norm.sf(c, NU[k] + rho * (GRID - MU[k]) / SD[k], np.sqrt(1 - rho ** 2)) for k in (0, 1)])
    pk = np.array([1 - PI, PI]); q = float(sum(pk[k] * norm.sf(c, NU[k], 1) for k in (0, 1)))
    return (pk[:, None] * f * a).sum(0) / q, (pk[:, None] * f * (1 - a)).sum(0) / (1 - q), q


def pop_raw_terms(p1, p0, q):
    D = p1 - p0; J = np.cumsum(D[::-1])[::-1]; i = int(np.argmax(J))
    return GRID[i], p1[i] / q + p0[i] / (1 - q), abs(np.gradient(D, DG)[i])        # t̃, Ṽ, Ã


def pop_smooth_sd(p1, p0, q, h, n):
    """Dạng chính xác (h cố định) của SD(t̂_h) và t_h, từ mật độ quần thể."""
    s = h / DG
    p1h, p0h = gaussian_filter1d(p1, s, truncate=8), gaussian_filter1d(p0, s, truncate=8)
    p1r, p0r = gaussian_filter1d(p1, s / np.sqrt(2), truncate=8), gaussian_filter1d(p0, s / np.sqrt(2), truncate=8)
    Dh = p1h - p0h; J = np.cumsum(Dh[::-1])[::-1]; i = int(np.argmax(J))
    v1 = p1r[i] * R / h - p1h[i] ** 2; v0 = p0r[i] * R / h - p0h[i] ** 2
    return float(np.sqrt((v1 / q + v0 / (1 - q)) / n) / abs(np.gradient(Dh, DG)[i])), GRID[i]


# ---------------------------------------------------------------- ước lượng không nhãn từ một mẫu
def estimators(f, pl, rng, boot=True):
    n = len(f); q = pl.mean(); h = bandwidth(f, pl); t = smooth_thr(f, pl, h=h)
    out = dict(h=h, t=t)
    try:
        V, A = kde_terms(np.sort(f), pl[np.argsort(f)])
    except Exception:
        V = A = np.nan
    out["G1_closed"] = np.sqrt(R * V / (n * h)) / A
    u = (f - t) / h; ph = norm.pdf(u) / h
    m1, m0 = ph[pl == 1].mean(), ph[pl == 0].mean()
    psi = np.where(pl == 1, (ph - m1) / q, -(ph - m0) / (1 - q))
    dD = (ph * u / h)[pl == 1].mean() - (ph * u / h)[pl == 0].mean()
    num = psi.std() / np.sqrt(n)
    out["G2_sandwich"] = num / abs(dD); out["G3_hybrid"] = num / A
    if boot:
        out["G4_boot"] = smooth_boot_sd(f, pl, rng)
    return out


def n_needed_smooth(V, A, c, m, eps):
    return (R * V * m ** 0.2 / (c * A ** 2 * eps ** 2)) ** 1.25


def n_needed_raw(V, A, eps):
    return 0.5134 ** 3 * 4 * V / (A ** 2 * eps ** 3)
