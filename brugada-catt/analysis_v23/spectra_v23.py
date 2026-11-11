"""Bước 23a: phổ của nhịp đại diện theo bộ dữ liệu và σ làm trơn khớp phổ (không dùng nhãn). Xem PLAN_vi.md.

Chạy: python analysis_v23/spectra_v23.py <harmonized> analysis_v23
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HD, OUT = sys.argv[1:3]
DOM = ["huca", "claris", "mortara"]
FREQ = np.fft.rfftfreq(75, d=0.01)                      # 38 tần số, 0–50 Hz
BAND = (FREQ >= 5) & (FREQ <= 40)
GRID_MS = np.arange(0, 31, 2)


def zs(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)


def smooth(X, sigma_ms):
    """Làm trơn Gauss theo thời gian cả 9 chuyển đạo, rồi z-score lại từng chuyển đạo."""
    if sigma_ms <= 0:
        return X.copy()
    return zs(gaussian_filter1d(X.astype(float), sigma_ms / 10.0, axis=-1, mode="nearest")).astype(np.float32)


def spectra(X, leads=(3, 4, 5)):
    x = X[:, list(leads)].astype(float); x = x - x.mean(-1, keepdims=True)
    p = np.abs(np.fft.rfft(x * np.hanning(75), axis=-1)) ** 2
    p = p / p.sum(-1, keepdims=True)
    return p.mean(1)                                     # (N, 38)


def hf(p):
    return p[:, FREQ >= 15].sum(1)


if __name__ == "__main__":
    Z = {n: np.load(os.path.join(HD, f"{n}.npz")) for n in DOM}
    X = {n: Z[n]["X"] for n in DOM}; Y = {n: Z[n]["y"].astype(int) for n in DOM}
    S = {"hf_v1v3": {}, "hf_9leads": {}, "hf_by_class": {}}
    rows = []
    for n in DOM:
        p = spectra(X[n]); h = hf(p); h9 = hf(spectra(X[n], range(9)))
        S["hf_v1v3"][n] = dict(median=round(float(np.median(h)), 4), iqr=[round(float(v), 4) for v in np.percentile(h, [25, 75])])
        S["hf_9leads"][n] = round(float(np.median(h9)), 4)
        S["hf_by_class"][n] = {int(c): round(float(np.median(h[Y[n] == c])), 4) for c in (0, 1)}
        med = np.median(p, 0)
        rows += [dict(dataset=n, freq_hz=round(float(f), 2), median_power=float(v)) for f, v in zip(FREQ, med)]
    ref = min(DOM, key=lambda n: S["hf_v1v3"][n]["median"]); S["reference"] = ref
    logref = np.log(np.median(spectra(X[ref]), 0)[BAND])
    S["sigma_ms"] = {ref: 0}; S["match_curve"] = {}
    for n in DOM:
        if n == ref:
            continue
        dist = [float(np.sqrt(np.mean((np.log(np.median(spectra(smooth(X[n], s)), 0)[BAND]) - logref) ** 2))) for s in GRID_MS]
        S["match_curve"][n] = {int(s): round(d, 4) for s, d in zip(GRID_MS, dist)}
        s_best = int(GRID_MS[int(np.argmin(dist))]); S["sigma_ms"][n] = s_best
        p = spectra(smooth(X[n], s_best))
        S["hf_v1v3"][f"{n}_matched"] = round(float(np.median(hf(p))), 4)
        rows += [dict(dataset=f"{n}_matched", freq_hz=round(float(f), 2), median_power=float(v)) for f, v in zip(FREQ, np.median(p, 0))]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "spectra_v23.csv"), index=False)
    json.dump(S, open(os.path.join(OUT, "spectra_v23.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))
