"""Bước 18a: kiểm hệ số quần thể hữu hạn (1 − n/N)^(1/3). Xem PLAN_vi.md.

Chạy: python analysis_v18/fpc_sim.py analysis_v18
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common18 import CONFIGS, curves, draw, subsample_masks, that  # noqa: E402

OUT = sys.argv[1]
rng = np.random.default_rng(1801)
NS = [400, 1200, 4000]; FR = [0.1, 0.2, 0.35, 0.5, 0.8]; POOLS = 200; SUBS = 200; IID = 2000
rows = []
for rho, shift in CONFIGS:
    sizes = sorted({int(round(fr * N)) for N in NS for fr in FR})
    sd_iid = {}
    for n in sizes:
        th = [that(*draw(rng, n, rho, shift))[0] for _ in range(IID)]
        th = np.array([t for t in th if t is not None]); sd_iid[n] = (float(th.std()), IID - len(th))
    for N in NS:
        var = {fr: [] for fr in FR}; msd = {fr: [] for fr in FR}; skipped = {fr: 0 for fr in FR}
        for _ in range(POOLS):
            f, pl = draw(rng, N, rho, shift); tN = that(f, pl)[0]
            for fr in FR:
                n = int(round(fr * N)); J, ok = curves(subsample_masks(rng, SUBS, N, n), pl)
                th = f[np.argmax(J, 1)][ok]; skipped[fr] += int((~ok).sum())
                var[fr].append(th.var()); msd[fr].append(np.mean((th - tN) ** 2))
        for fr in FR:
            n = int(round(fr * N)); sd_pool = float(np.sqrt(np.mean(var[fr])))
            rows.append(dict(rho=rho, shift=shift, N=N, frac=fr, n=n, sd_pool=sd_pool, sd_iid=sd_iid[n][0],
                             rms_to_tN=float(np.sqrt(np.mean(msd[fr]))), R=sd_pool / sd_iid[n][0],
                             pred=(1 - fr) ** (1 / 3), skipped_sub=skipped[fr], skipped_iid=sd_iid[n][1]))
        print("done", rho, shift, N, flush=True)
D = pd.DataFrame(rows); D["R_over_pred"] = D.R / D.pred
D.round(5).to_csv(os.path.join(OUT, "fpc_sim.csv"), index=False)
print(D.groupby("frac")[["R", "pred", "R_over_pred"]].median().round(3))
