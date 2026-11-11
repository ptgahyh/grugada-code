"""Bước 20, S1: ngưỡng làm trơn so với ngưỡng thường trong mô phỏng Gauss. Chạy: python analysis_v20/smooth_sim.py analysis_v20"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "analysis_v18"))
from common18 import CONFIGS, MID, MU, NU, PI, SD, draw  # noqa: E402
from smooth_lib import raw_thr, smooth_boot_sd, smooth_thr  # noqa: E402

OUT = sys.argv[1]
rng = np.random.default_rng(2001)
GRID = np.linspace(-4, 7, 22001); DT = GRID[1] - GRID[0]
ba = lambda t: 0.5 * (norm.sf(t, MU[1], SD[1]) + norm.cdf(t, MU[0], SD[0]))
BEST = ba(GRID).max()


def t_tilde(rho, c):
    """Ngưỡng CATT quần thể (như analysis_v13/unified_sim.population)."""
    f = np.stack([norm.pdf(GRID, MU[k], SD[k]) for k in (0, 1)])
    a = np.stack([norm.sf(c, NU[k] + rho * (GRID - MU[k]) / SD[k], np.sqrt(1 - rho ** 2)) for k in (0, 1)])
    pk = np.array([1 - PI, PI]); q = float(sum(pk[k] * norm.sf(c, NU[k], 1) for k in (0, 1)))
    d1 = (pk[:, None] * f * a).sum(0) / q; d0 = (pk[:, None] * f * (1 - a)).sum(0) / (1 - q)
    return GRID[int(np.argmax(np.cumsum((d1 - d0)[::-1])[::-1]))]


rows, brows = [], []
for rho, shift in CONFIGS:
    tt = t_tilde(rho, MID + shift)
    for n in (100, 300, 1000, 3000):
        T = {k: [] for k in ("raw", "s1.0", "s0.5", "s2.0")}
        for r in range(2000):
            f, pl = draw(rng, n, rho, shift)
            if pl.sum() < 3 or n - pl.sum() < 3:
                continue
            T["raw"].append(raw_thr(f, pl))
            for m in (1.0, 0.5, 2.0):
                T[f"s{m}"].append(smooth_thr(f, pl, m))
            if r < 200:
                brows.append(dict(rho=rho, shift=shift, n=n, boot_sd=smooth_boot_sd(f, pl, rng)))
        row = dict(rho=rho, shift=shift, n=n, t_tilde=tt, used=len(T["raw"]))
        for k, v in T.items():
            v = np.array(v)
            row.update({f"rmse_{k}": float(np.sqrt(np.mean((v - tt) ** 2))), f"sd_{k}": float(v.std()),
                        f"bias_{k}": float(v.mean() - tt), f"baloss_{k}": float(np.mean(BEST - ba(v)))})
        rows.append(row)
        for b in brows[-200:]:
            b["sd_true"] = row["sd_s1.0"]
        print("done", rho, shift, n, round(row["rmse_s1.0"] / row["rmse_raw"], 3), round(row["baloss_s1.0"] / row["baloss_raw"], 3), flush=True)
pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "smooth_sim.csv"), index=False)
pd.DataFrame(brows).round(6).to_csv(os.path.join(OUT, "smooth_sim_boot.csv"), index=False)
