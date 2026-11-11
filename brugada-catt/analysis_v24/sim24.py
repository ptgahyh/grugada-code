"""Bước 24, T1 + T2: công thức với đại lượng quần thể, và ước lượng không nhãn từ một mẫu. Chạy: python analysis_v24/sim24.py analysis_v24"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib24 import R, bandwidth, estimators, pop_densities, pop_raw_terms, pop_smooth_sd, smooth_thr  # noqa: E402
from common18 import CONFIGS, MID, draw  # noqa: E402

OUT = sys.argv[1]; rng = np.random.default_rng(2401)
cells, rows = [], []
for rho, shift in CONFIGS:
    p1, p0, q = pop_densities(rho, MID + shift); tt, V, A = pop_raw_terms(p1, p0, q)
    for n in (100, 300, 1000, 3000):
        th, hs = [], []
        for r in range(2000):
            f, pl = draw(rng, n, rho, shift)
            if pl.sum() < 3 or n - pl.sum() < 3:
                continue
            h = bandwidth(f, pl); th.append(smooth_thr(f, pl, h=h)); hs.append(h)
            if r < 200:
                rows.append(dict(rho=rho, shift=shift, n=n, **estimators(f, pl, rng)))
        th = np.array(th); hm = float(np.median(hs)); sd_exact, t_h = pop_smooth_sd(p1, p0, q, hm, n)
        cells.append(dict(rho=rho, shift=shift, n=n, used=len(th), h_median=hm, sd_obs=float(th.std()),
                          pred_exact=sd_exact, pred_closed=float(np.sqrt(R * V / (n * hm)) / A), t_tilde=tt, t_h=t_h,
                          mean_t=float(th.mean()), V=V, A=A, q=q))
        print("done", rho, shift, n, round(sd_exact / th.std(), 3), flush=True)
C = pd.DataFrame(cells); C.round(6).to_csv(os.path.join(OUT, "sim24_cells.csv"), index=False)
D = pd.DataFrame(rows).merge(C[["rho", "shift", "n", "sd_obs"]], on=["rho", "shift", "n"])
D.round(6).to_csv(os.path.join(OUT, "sim24_estimators.csv"), index=False)
