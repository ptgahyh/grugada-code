"""So sánh lại công bằng sau phát hiện: CNN chỉ dùng V1–V3 chuyển miền tốt hơn hẳn CNN 9 chuyển đạo.

Mô hình (9 thiết lập ngoài miền, 3 seed chia validation như run_v2.py):
  CNN-V1V3        CNN 1D trên V1, V2, V3 (ensemble 3 seed khởi tạo)
  CBM2-SignAda    như v2 (huấn luyện lại để có điểm validation cho ngưỡng/hợp nhất)
  CBM2-SignAda-noShift  (chỉ LODO) như SignAda nhưng bỏ thêm khái niệm có dịch chuyển lớn giữa các miền
                  TRAIN trong nhóm âm tính (|SMD| > 0.5) — chỉ dùng dữ liệu train
  Hybrid          trung bình xác suất của CNN-V1V3 và CBM2-SignAda; ngưỡng chọn trên validation
Điểm của CNN (9 chuyển đạo) và ST_V2 lấy lại từ concepts_v2/v2_raw.json (cùng seed, cùng cách chia).

Chạy: python compare_v3.py <harmonized> <concepts_v2> <output>
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbm_lib import (DOMAINS, K, cbm_predict, cnn_predict, concepts_fixed, load, pack, sign_gate,  # noqa: E402
                     strat_val, train_cbm2, train_cnn, youden)

HD, V2, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
X, Y = load(HD)
SETTINGS = [([a], b) for a in DOMAINS for b in DOMAINS if a != b] + [([a for a in DOMAINS if a != b], b) for b in DOMAINS]


def rank01(s):
    return (np.argsort(np.argsort(s)) + 0.5) / len(s)


def shift_mask(xtr, ytr, dtr, tau=0.5):
    C = concepts_fixed(xtr); doms = np.unique(dtr); keep = np.ones(K, bool)
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            a = C[(dtr == doms[i]) & (ytr == 0)]; b = C[(dtr == doms[j]) & (ytr == 0)]
            smd = (b.mean(0) - a.mean(0)) / np.sqrt((a.var(0) + b.var(0)) / 2 + 1e-12)
            keep &= np.abs(smd) <= tau
    return keep


rows = []
t0 = time.time()
for src, tgt in SETTINGS:
    xtr, ytr, dtr, di = pack(X, Y, src)
    for vs in (0, 1, 2):
        tr, va = strat_val(ytr, di, vs)
        out = {}
        # CNN-V1V3
        nets = [train_cnn(xtr[tr][:, 3:6], ytr[tr], xtr[va][:, 3:6], ytr[va], s) for s in (0, 1, 2)]
        sv_c, st_c = cnn_predict(nets, xtr[va][:, 3:6]), cnn_predict(nets, X[tgt][:, 3:6])
        out["CNN-V1V3"] = (st_c, youden(ytr[va], sv_c), None)
        # CBM2-SignAda
        signs = sign_gate(xtr[tr], ytr[tr], dtr[tr])
        ms = [train_cbm2(xtr[tr], ytr[tr], dtr[tr], xtr[va], ytr[va], dtr[va], s, True, signs) for s in (0, 1)]
        sv_b, st_b = cbm_predict(ms, xtr[va], True, dom=dtr[va]), cbm_predict(ms, X[tgt], True)
        out["CBM2-SignAda"] = (st_b, youden(ytr[va], sv_b), int((signs != 0).sum()))
        # Hybrid: trung bình xác suất của hai mô hình (không xếp hạng trong lô, để ngưỡng không phụ thuộc tỉ lệ bệnh)
        sv_h, st_h = (sv_c + sv_b) / 2, (st_c + st_b) / 2
        out["Hybrid"] = (st_h, youden(ytr[va], sv_h), None)
        # LODO: bỏ thêm khái niệm dịch chuyển mạnh giữa các miền train
        if len(src) > 1:
            keep = shift_mask(xtr[tr], ytr[tr], dtr[tr])
            s2 = signs * keep
            ms2 = [train_cbm2(xtr[tr], ytr[tr], dtr[tr], xtr[va], ytr[va], dtr[va], s, True, s2) for s in (0, 1)]
            sv2, st2 = cbm_predict(ms2, xtr[va], True, dom=dtr[va]), cbm_predict(ms2, X[tgt], True)
            out["CBM2-SignAda-noShift"] = (st2, youden(ytr[va], sv2), int((s2 != 0).sum()))
        y = Y[tgt]
        for m, (s, thr, nk) in out.items():
            p = s >= thr
            rows.append(dict(train="+".join(src), test=tgt, model=m, vseed=vs, auroc=round(roc_auc_score(y, s), 4),
                             sens=round(float(p[y == 1].mean()), 3), spec=round(float((~p)[y == 0].mean()), 3),
                             n_kept=nk, scores=s.round(5).tolist()))
            print({k: rows[-1][k] for k in ["train", "test", "model", "vseed", "auroc", "sens", "spec", "n_kept"]}, f"{time.time() - t0:.0f}s", flush=True)

# thêm CNN (9 chuyển đạo) và ST_V2 từ v2 để phân tích chung
for r in json.load(open(os.path.join(V2, "v2_raw.json"))):
    if r["model"] in ("CNN", "ST_V2"):
        rows.append({k: r[k] for k in ["train", "test", "model", "vseed", "auroc", "sens", "spec", "scores"]} | {"n_kept": None})
json.dump(rows, open(os.path.join(OUT, "v3_raw.json"), "w"))
print("xong", f"{time.time() - t0:.0f}s")
