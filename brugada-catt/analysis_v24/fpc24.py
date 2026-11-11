"""Bước 24, T3: số mũ hệ số quần thể hữu hạn cho ngưỡng làm trơn (như bước 18a). Chạy: python analysis_v24/fpc24.py analysis_v24"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib24 import smooth_thr  # noqa: E402
from common18 import CONFIGS, draw  # noqa: E402

OUT = sys.argv[1]; rng = np.random.default_rng(2403)
NS = [400, 1200, 4000]; FR = [0.1, 0.2, 0.35, 0.5, 0.8]; POOLS = 100; SUBS = 200; IID = 2000


def thr(f, pl):
    return smooth_thr(f, pl) if 3 <= pl.sum() <= len(pl) - 3 else None


rows = []
for rho, shift in CONFIGS:
    sizes = sorted({int(round(fr * N)) for N in NS for fr in FR}); sd_iid = {}
    for n in sizes:
        th = [t for t in (thr(*draw(rng, n, rho, shift)) for _ in range(IID)) if t is not None]; sd_iid[n] = float(np.std(th))
    for N in NS:
        var = {fr: [] for fr in FR}
        for _ in range(POOLS):
            f, pl = draw(rng, N, rho, shift)
            for fr in FR:
                n = int(round(fr * N)); th = []
                for _ in range(SUBS):
                    k = rng.choice(N, n, replace=False); t = thr(f[k], pl[k])
                    if t is not None:
                        th.append(t)
                var[fr].append(np.var(th))
        for fr in FR:
            n = int(round(fr * N)); R_ = float(np.sqrt(np.mean(var[fr]))) / sd_iid[n]
            rows.append(dict(rho=rho, shift=shift, N=N, frac=fr, n=n, sd_iid=sd_iid[n], R=R_))
        print("done", rho, shift, N, flush=True)
pd.DataFrame(rows).round(5).to_csv(os.path.join(OUT, "fpc24.csv"), index=False)
