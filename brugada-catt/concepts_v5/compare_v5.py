"""Bước 5: thay QRS_w bằng QRS_sd15, và lưu điểm validation để hiệu chỉnh ngưỡng không nhãn (recal_v5.py).

Mô hình (9 thiết lập ngoài miền, 3 seed chia validation, giống compare_v4):
  CNN-V1V3       ensemble 3 seed (cùng seed -> phải trùng compare_v4)
  CBM3-Sign      khái niệm v3 (QRS_w), cổng dấu (-> phải trùng compare_v4)
  CBM3n-Sign     khái niệm v3, BỎ khái niệm độ rộng QRS (ép dấu = 0 -> trọng số = 0)
  CBM5-Sign      khái niệm v5 (QRS_sd15), cổng dấu
  CBM5-SignAda   khái niệm v5, chuẩn hóa theo miền
  Hybrid-S3 / S3n / S5 / A5   trung bình xác suất CNN-V1V3 + CBM tương ứng
Mỗi dòng lưu: nhãn + điểm + miền của tập validation (nguồn), điểm trên miền đích.

Chạy: python compare_v5.py <harmonized> <output> <chunk 0|1> <số chunk>
Kết quả ghi dần vào v5_raw_<chunk>.jsonl (mỗi dòng một (thiết lập, seed validation)).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
sys.path.insert(0, HERE)
import cbm_lib4 as L  # noqa: E402
import concepts5  # noqa: E402

torch.set_num_threads(1)
L.CLS["v5"] = lambda learn=True: concepts5.BrugadaConcepts(learn=learn, qrs="sd15")
L.FROZEN["v5"] = L.CLS["v5"](learn=False)

HD, OUT, CH, NCH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
X, Y = L.load(HD)
DOM = L.DOMAINS
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
JOBS = [(i, vs) for i in range(len(SETTINGS)) for vs in (0, 1, 2)]
JOBS = JOBS[CH::NCH]
fout = os.path.join(OUT, f"v5_raw_{CH}.jsonl")
done = set()
if os.path.exists(fout):
    for line in open(fout):
        r = json.loads(line); done.add((r["setting"], r["vseed"]))
t0 = time.time()
for i, vs in JOBS:
    if (i, vs) in done:
        continue
    src, tgt = SETTINGS[i]
    xtr, ytr, dtr, di = L.pack(X, Y, src)
    tr, va = L.strat_val(ytr, di, vs)
    P, SIGNS = {}, {}
    nets = [L.train_cnn(xtr[tr][:, 3:6], ytr[tr], xtr[va][:, 3:6], ytr[va], s) for s in (0, 1, 2)]
    P["CNN-V1V3"] = (L.cnn_predict(nets, xtr[va][:, 3:6]), L.cnn_predict(nets, X[tgt][:, 3:6]))
    for name, cv, ada, drop in [("CBM3-Sign", "v3", False, False), ("CBM3n-Sign", "v3", False, True),
                                ("CBM5-Sign", "v5", False, False), ("CBM5-SignAda", "v5", True, False)]:
        signs = L.sign_gate(xtr[tr], ytr[tr], dtr[tr], cv=cv)
        if drop:
            signs[-1] = 0.0
        SIGNS[name] = signs.tolist()
        ms = [L.train_cbm2(xtr[tr], ytr[tr], dtr[tr], xtr[va], ytr[va], dtr[va], s, ada, signs, cv=cv) for s in (0, 1)]
        P[name] = (L.cbm_predict(ms, xtr[va], ada, dom=dtr[va] if ada else None), L.cbm_predict(ms, X[tgt], ada))
    for h, b in [("Hybrid-S3", "CBM3-Sign"), ("Hybrid-S3n", "CBM3n-Sign"), ("Hybrid-S5", "CBM5-Sign"), ("Hybrid-A5", "CBM5-SignAda")]:
        P[h] = ((P["CNN-V1V3"][0] + P[b][0]) / 2, (P["CNN-V1V3"][1] + P[b][1]) / 2)
    rec = dict(setting=i, train="+".join(src), test=tgt, vseed=vs, y_val=ytr[va].tolist(), d_val=dtr[va].tolist(), signs=SIGNS,
               models={m: dict(val=sv.round(6).tolist(), tgt=st.round(6).tolist(),
                               auroc=round(float(roc_auc_score(Y[tgt], st)), 4)) for m, (sv, st) in P.items()})
    with open(fout, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(i, "+".join(src), "->", tgt, "vseed", vs, {m: rec["models"][m]["auroc"] for m in P}, f"{time.time() - t0:.0f}s", flush=True)
print("xong", f"{time.time() - t0:.0f}s")
