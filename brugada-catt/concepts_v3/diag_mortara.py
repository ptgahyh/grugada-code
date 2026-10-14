"""Chẩn đoán vì sao Mortara khó cho mô hình khái niệm.

1. Dịch chuyển khái niệm Claris -> Mortara (cùng bệnh viện, khác máy): chênh lệch chuẩn hóa (SMD)
   của từng khái niệm, tách riêng nhóm âm tính và nhóm Brugada; và AUROC đơn biến trong từng miền.
2. CNN bị giới hạn thông tin (huấn luyện + test cùng cách giới hạn), 9 thiết lập ngoài miền:
     CNN-V1V3   chỉ V1, V2, V3
     CNN-noST   9 chuyển đạo nhưng xóa đoạn ST-T (từ 60 ms sau đỉnh QRS trở đi -> thay bằng giá trị tại mốc)
     CNN-QRS    chỉ giữ cửa sổ QRS (−90..+60 ms), phần còn lại xóa
   So với CNN đủ 9 chuyển đạo (điểm từ concepts_v2, seed validation 0).
3. Phân tích lỗi ở HUCA+Claris -> Mortara: ca Brugada CNN bắt được mà CBM2-SignAda bỏ sót.

Chạy: python diag_mortara.py <harmonized> <concepts_v2> <output>
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbm_lib import (CONCEPT_NAMES, DOMAINS, cnn_predict, concepts_fixed, load, pack, strat_val,  # noqa: E402
                     train_cnn, youden)

HD, V2, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
X, Y = load(HD)
ANCHOR = 28
out = {}

# ------------------------------------------------------------------ 1. dịch chuyển khái niệm
Cc, Cm = concepts_fixed(X["claris"]), concepts_fixed(X["mortara"])
shift = []
for k, n in enumerate(CONCEPT_NAMES):
    r = dict(concept=n)
    for lab, nm in [(0, "neg"), (1, "pos")]:
        a, b = Cc[Y["claris"] == lab, k], Cm[Y["mortara"] == lab, k]
        r[f"smd_{nm}"] = round(float((b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)), 3)
    r["auc_claris"] = round(roc_auc_score(Y["claris"], Cc[:, k]), 3)
    r["auc_mortara"] = round(roc_auc_score(Y["mortara"], Cm[:, k]), 3)
    shift.append(r)
out["concept_shift_claris_to_mortara"] = sorted(shift, key=lambda r: -abs(r["smd_neg"]))

# ------------------------------------------------------------------ 2. CNN bị giới hạn thông tin
def restrict(x, mode):
    x = x.copy()
    if mode == "V1V3":
        return x[:, 3:6]
    if mode == "noST":
        x[:, :, ANCHOR + 6:] = x[:, :, ANCHOR + 6:ANCHOR + 7]
        return x
    if mode == "QRS":
        x[:, :, :ANCHOR - 9] = x[:, :, ANCHOR - 9:ANCHOR - 8]
        x[:, :, ANCHOR + 6:] = x[:, :, ANCHOR + 6:ANCHOR + 7]
        return x


raw2 = json.load(open(os.path.join(V2, "v2_raw.json")))
full = {(r["train"], r["test"]): np.array(r["scores"]) for r in raw2 if r["model"] == "CNN" and r["vseed"] == 0}
SETTINGS = [([a], b) for a in DOMAINS for b in DOMAINS if a != b] + [([a for a in DOMAINS if a != b], b) for b in DOMAINS]
abl = []
t0 = time.time()
for src, tgt in SETTINGS:
    xtr, ytr, dtr, di = pack(X, Y, src)
    tr, va = strat_val(ytr, di, 0)
    r = dict(train="+".join(src), test=tgt, CNN=round(roc_auc_score(Y[tgt], full[("+".join(src), tgt)]), 3))
    for mode in ["V1V3", "noST", "QRS"]:
        xr, xt = restrict(xtr, mode), restrict(X[tgt], mode)
        nets = [train_cnn(xr[tr], ytr[tr], xr[va], ytr[va], s) for s in (0, 1, 2)]
        r[f"CNN-{mode}"] = round(roc_auc_score(Y[tgt], cnn_predict(nets, xt)), 3)
    abl.append(r); print(r, f"{time.time() - t0:.0f}s", flush=True)
out["cnn_restriction"] = abl

# ------------------------------------------------------------------ 3. phân tích lỗi HUCA+Claris -> Mortara
def avg_scores(model, tr="huca+claris", te="mortara"):
    return np.mean([np.array(r["scores"]) for r in raw2 if r["model"] == model and r["train"] == tr and r["test"] == te], 0)


s_cnn, s_cbm = avg_scores("CNN"), avg_scores("CBM2-SignAda")
y = Y["mortara"]
# hạng phần trăm trong miền test (không phụ thuộc thang điểm)
rk = lambda s: (np.argsort(np.argsort(s)) + 0.5) / len(s)
r_cnn, r_cbm = rk(s_cnn), rk(s_cbm)
q = np.quantile(r_cnn[y == 0], 0.8)                       # ngưỡng: 80% ca âm tính nằm dưới
qc = np.quantile(r_cbm[y == 0], 0.8)
groups = {"ca_dương_cả_hai_bắt": (y == 1) & (r_cnn > q) & (r_cbm > qc),
          "chỉ_CNN_bắt": (y == 1) & (r_cnn > q) & (r_cbm <= qc),
          "chỉ_CBM_bắt": (y == 1) & (r_cnn <= q) & (r_cbm > qc),
          "cả_hai_bỏ_sót": (y == 1) & (r_cnn <= q) & (r_cbm <= qc),
          "âm_tính": y == 0}
Cm = concepts_fixed(X["mortara"])
key = ["J_elev_V1", "J_elev_V2", "Corrado_V1", "Corrado_V2", "slope_V2", "T_V1", "T_V2", "r_prime_V2", "QRS_w"]
err = {}
for g, m in groups.items():
    err[g] = dict(n=int(m.sum()), **{k: round(float(np.median(Cm[m, CONCEPT_NAMES.index(k)])), 3) for k in key})
out["mortara_error_groups"] = err
np.savez_compressed(os.path.join(OUT, "mortara_groups.npz"), **{g: X["mortara"][m] for g, m in groups.items()})
json.dump(out, open(os.path.join(OUT, "diag_mortara.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(err, indent=1, ensure_ascii=False))
print("xong", f"{time.time() - t0:.0f}s")
