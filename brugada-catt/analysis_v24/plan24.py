"""Bước 24, T4: lập kế hoạch cỡ mẫu cho ngưỡng làm trơn từ mẫu thử. Chạy: python analysis_v24/plan24.py analysis_v24"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib24 import c_const, kde_terms, n_needed_raw, n_needed_smooth, pop_densities, pop_raw_terms, smooth_thr  # noqa: E402
from common18 import CONFIGS, MID, draw  # noqa: E402

OUT = sys.argv[1]; rng = np.random.default_rng(2404)
NGRID = np.unique(np.round(np.logspace(np.log10(50), np.log10(30000), 30)).astype(int)); EPS = (0.15, 0.30)
tab, plans, desc = [], [], []
for rho, shift in CONFIGS:
    sd = []
    for n in NGRID:
        th = []
        for _ in range(1000):
            f, pl = draw(rng, n, rho, shift)
            if 3 <= pl.sum() <= n - 3:
                th.append(smooth_thr(f, pl))
        sd.append(np.std(th)); tab.append(dict(rho=rho, shift=shift, n=int(n), sd=float(np.std(th)), bias_mean_t=float(np.mean(th))))
    lx, ly = np.log(NGRID), np.log(sd); slope = np.polyfit(lx[-5:], ly[-5:], 1)[0]
    sd_at = lambda n: float(np.exp(np.interp(np.log(n), lx, ly) if n <= NGRID[-1] else ly[-1] + slope * (np.log(n) - lx[-1])))
    # hằng số quần thể (mô tả)
    p1, p0, q = pop_densities(rho, MID + shift); tt, V, A = pop_raw_terms(p1, p0, q)
    fb, plb = draw(rng, 400000, rho, shift); c = c_const(fb); m = min(q, 1 - q)
    for e in EPS:
        desc.append(dict(rho=rho, shift=shift, eps=e, n_smooth_pop=n_needed_smooth(V, A, c, m, e), n_raw_pop=n_needed_raw(V, A, e)))
    for mpil in (100, 300):
        for _ in range(200):
            f, pl = draw(rng, mpil, rho, shift)
            if not 3 <= pl.sum() <= mpil - 3:
                continue
            V_, A_ = kde_terms(f, pl); q_ = pl.mean()
            for e in EPS:
                n_p = max(n_needed_smooth(V_, A_, c_const(f), min(q_, 1 - q_), e), 20.0)
                plans.append(dict(rho=rho, shift=shift, m=mpil, eps=e, n_plan=n_p, sd_true=sd_at(n_p), ok=sd_at(n_p) <= 1.2 * e))
    print("done", rho, shift, flush=True)
pd.DataFrame(tab).round(5).to_csv(os.path.join(OUT, "plan24_table.csv"), index=False)
pd.DataFrame(plans).round(5).to_csv(os.path.join(OUT, "plan24_plans.csv"), index=False)
pd.DataFrame(desc).round(2).to_csv(os.path.join(OUT, "plan24_pop_n.csv"), index=False)
