"""Bước 18b: F1 nửa mẫu so với các ước lượng có chứng minh, trong mô phỏng biết SD thật. Xem PLAN_vi.md.

Chạy: python analysis_v18/est_sim.py analysis_v18
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common18 import CONFIGS, draw, estimators, that  # noqa: E402

OUT = sys.argv[1]
rng = np.random.default_rng(1802)
NS = [100, 300, 1000, 3000]; TRUE = 2000; SAMPLES = 200
rows = []
for rho, shift in CONFIGS:
    for n in NS:
        th = [that(*draw(rng, n, rho, shift))[0] for _ in range(TRUE)]
        th = np.array([t for t in th if t is not None]); sd_true = float(th.std())
        for s in range(SAMPLES):
            f, pl = draw(rng, n, rho, shift); e = estimators(f, pl, rng)
            if e is None:
                rows.append(dict(rho=rho, shift=shift, n=n, sample=s, sd_true=sd_true, skipped=1)); continue
            rows.append(dict(rho=rho, shift=shift, n=n, sample=s, sd_true=sd_true, skipped=0, **e))
        print("done", rho, shift, n, round(sd_true, 4), flush=True)
pd.DataFrame(rows).round(6).to_csv(os.path.join(OUT, "est_sim.csv"), index=False)
