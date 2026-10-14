"""Bước 4: khái niệm chỉ từ V1–V3 (v3) và Hybrid KHÔNG thích nghi (dùng được khi tỉ lệ bệnh thấp).

Mô hình (9 thiết lập ngoài miền, 3 seed chia validation, giống run_v2/compare_v3):
  CNN-V1V3       huấn luyện lại (cùng seed -> phải trùng compare_v3)
  CBM2-Sign      khái niệm v2, không thích nghi
  CBM3-Sign      khái niệm v3 (điểm J, QRS_w chỉ từ V1–V3), không thích nghi
  CBM3-SignAda   khái niệm v3, chuẩn hóa theo miền
  Hybrid-S2      trung bình xác suất CNN-V1V3 + CBM2-Sign
  Hybrid-S3      trung bình xác suất CNN-V1V3 + CBM3-Sign
  Hybrid-A3      trung bình xác suất CNN-V1V3 + CBM3-SignAda
Lưu kèm điểm validation để phân tích ngưỡng.

Chạy: python compare_v4.py <harmonized> <output>
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbm_lib4 import (DOMAINS, cbm_predict, cnn_predict, load, pack, sign_gate, strat_val, train_cbm2,  # noqa: E402
                      train_cnn, youden)

HD, OUT = sys.argv[1], sys.argv[2]
X, Y = load(HD)
SETTINGS = [([a], b) for a in DOMAINS for b in DOMAINS if a != b] + [([a for a in DOMAINS if a != b], b) for b in DOMAINS]
rows = []
t0 = time.time()
for src, tgt in SETTINGS:
    xtr, ytr, dtr, di = pack(X, Y, src)
    for vs in (0, 1, 2):
        tr, va = strat_val(ytr, di, vs)
        P = {}
        nets = [train_cnn(xtr[tr][:, 3:6], ytr[tr], xtr[va][:, 3:6], ytr[va], s) for s in (0, 1, 2)]
        P["CNN-V1V3"] = (cnn_predict(nets, xtr[va][:, 3:6]), cnn_predict(nets, X[tgt][:, 3:6]))
        for name, cv, ada in [("CBM2-Sign", "v2", False), ("CBM3-Sign", "v3", False), ("CBM3-SignAda", "v3", True)]:
            signs = sign_gate(xtr[tr], ytr[tr], dtr[tr], cv=cv)
            ms = [train_cbm2(xtr[tr], ytr[tr], dtr[tr], xtr[va], ytr[va], dtr[va], s, ada, signs, cv=cv) for s in (0, 1)]
            P[name] = (cbm_predict(ms, xtr[va], ada, dom=dtr[va] if ada else None), cbm_predict(ms, X[tgt], ada))
        for h, b in [("Hybrid-S2", "CBM2-Sign"), ("Hybrid-S3", "CBM3-Sign"), ("Hybrid-A3", "CBM3-SignAda")]:
            P[h] = ((P["CNN-V1V3"][0] + P[b][0]) / 2, (P["CNN-V1V3"][1] + P[b][1]) / 2)
        y = Y[tgt]
        for m, (sv, st) in P.items():
            thr = youden(ytr[va], sv); p = st >= thr
            rows.append(dict(train="+".join(src), test=tgt, model=m, vseed=vs, auroc=round(roc_auc_score(y, st), 4),
                             sens=round(float(p[y == 1].mean()), 3), spec=round(float((~p)[y == 0].mean()), 3),
                             thr=float(thr), scores=st.round(5).tolist()))
            print({k: rows[-1][k] for k in ["train", "test", "model", "vseed", "auroc", "sens", "spec"]}, f"{time.time() - t0:.0f}s", flush=True)
json.dump(rows, open(os.path.join(OUT, "v4_raw.json"), "w"))
print("xong", f"{time.time() - t0:.0f}s")
