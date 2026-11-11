"""Bước 21a/21b: huấn luyện lại CNN-V1V3 như bước 8; chấm điểm đích khi dịch thời gian; lặp lại với cách căn chung theo đáy S của V1.

Chạy: python analysis_v21/train_v21.py <harmonized> <thư mục riêng> [chunk] [số chunk]
Ghi vào <thư mục riêng>: scores_{orig|realign}_{thiết lập}_{seed}.npz và mô hình một nguồn (seed 0) cho 21c.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
import cbm_lib4 as L  # noqa: E402

torch.set_num_threads(1)
HD, PRIV = sys.argv[1:3]
CH, NCH = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
SHIFTS = [-3, -2, -1, 0, 1, 2, 3]
DOM = L.DOMAINS
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]


def shift(x, s):
    """Dịch theo trục thời gian s mẫu (s > 0: trễ hơn), đệm bằng giá trị biên. x: (..., T)."""
    if s == 0:
        return x
    out = np.empty_like(x)
    if s > 0:
        out[..., s:] = x[..., :-s]; out[..., :s] = x[..., :1]
    else:
        out[..., :s] = x[..., -s:]; out[..., s:] = x[..., -1:]
    return out


def realign(X):
    """Đặt đáy sóng S của V1 (cực tiểu trong mẫu 18–39) tại mẫu 28 cho từng bản ghi."""
    nad = np.argmin(X[:, 3, 18:40], 1) + 18
    return np.stack([shift(x, int(28 - k)) for x, k in zip(X, nad)])


if __name__ == "__main__":
    X0, Y = L.load(HD)
    VAR = {"orig": X0, "realign": {n: realign(X0[n]) for n in DOM}}
    os.makedirs(PRIV, exist_ok=True); t0 = time.time()
    JOBS = [(v, i, vs) for v in VAR for i in range(9) for vs in (0, 1, 2)][CH::NCH]
    for v, i, vs in JOBS:
        fn = os.path.join(PRIV, f"scores_{v}_{i}_{vs}.npz")
        if os.path.exists(fn):
            continue
        X = VAR[v]; src, tgt = SETTINGS[i]
        xtr, ytr, _, di = L.pack(X, Y, src); tr, va = L.strat_val(ytr, di, vs)
        xv, yv, xt, yt = xtr[va][:, 3:6], ytr[va], X[tgt][:, 3:6], Y[tgt]
        nets = [L.train_cnn(xtr[tr][:, 3:6], ytr[tr], xv, yv, s) for s in (0, 1, 2)]
        out = dict(y_val=yv, y_tgt=yt, cnn_val=L.cnn_predict(nets, xv), cnn_tgt=L.cnn_predict(nets, xt))
        if v == "orig":
            for s in SHIFTS:
                out[f"cnn_tgt_shift{s}"] = L.cnn_predict(nets, np.ascontiguousarray(shift(xt, s)))
            if len(src) == 1 and vs == 0:
                torch.save([n.state_dict() for n in nets], os.path.join(PRIV, f"model_{src[0]}.pt"))
        np.savez_compressed(fn, **out)
        print(v, i, "+".join(src), "->", tgt, vs, f"{time.time() - t0:.0f}s", flush=True)
    print("xong", flush=True)
