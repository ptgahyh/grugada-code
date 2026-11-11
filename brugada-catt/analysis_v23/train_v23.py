"""Bước 23b–23d: chấm điểm lại sau khi đồng bộ dải thông, và huấn luyện lại với dải thông đồng bộ. Xem PLAN_vi.md.

Chạy: python analysis_v23/train_v23.py <harmonized> <analysis_v23/spectra_v23.json> <thư mục riêng> [chunk] [số chunk]
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4")); sys.path.insert(0, HERE)
import cbm_lib4 as L  # noqa: E402
from spectra_v23 import smooth  # noqa: E402

torch.set_num_threads(1)
HD, SPEC, PRIV = sys.argv[1:4]
CH, NCH = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
SIG = json.load(open(SPEC))["sigma_ms"]
DOM = L.DOMAINS
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
FACT = [0.5, 1.0, 1.5, 2.0]

X0, Y = L.load(HD)
VAR = {"orig": X0, "matched": {n: smooth(X0[n], SIG[n]) for n in DOM}, "common20": {n: smooth(X0[n], 20) for n in DOM}}
TT = {k: {n: smooth(X0[n], k * SIG[n]) for n in DOM} for k in FACT}          # chấm điểm lại, cho 23b
os.makedirs(PRIV, exist_ok=True); t0 = time.time()
for v, i, vs in [(v, i, vs) for v in VAR for i in range(9) for vs in (0, 1, 2)][CH::NCH]:
    fn = os.path.join(PRIV, f"scores_{v}_{i}_{vs}.npz")
    if os.path.exists(fn):
        continue
    X = VAR[v]; src, tgt = SETTINGS[i]
    xtr, ytr, _, di = L.pack(X, Y, src); tr, va = L.strat_val(ytr, di, vs)
    xv, yv, xt, yt = xtr[va][:, 3:6], ytr[va], X[tgt][:, 3:6], Y[tgt]
    nets = [L.train_cnn(xtr[tr][:, 3:6], ytr[tr], xv, yv, s) for s in (0, 1, 2)]
    out = dict(y_val=yv, y_tgt=yt, cnn_val=L.cnn_predict(nets, xv), cnn_tgt=L.cnn_predict(nets, xt))
    if v == "orig":
        for k in FACT:
            xs_k, _, _, _ = L.pack(TT[k], Y, src)
            out[f"cnn_val_s{k}"] = L.cnn_predict(nets, xs_k[va][:, 3:6]); out[f"cnn_tgt_s{k}"] = L.cnn_predict(nets, TT[k][tgt][:, 3:6])
    np.savez_compressed(fn, **out)
    print(v, i, "+".join(src), "->", tgt, vs, f"{time.time() - t0:.0f}s", flush=True)
print("xong", flush=True)
