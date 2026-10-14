"""(Bước 8) Diễn giải CBM: mô hình đã học gì, và khái niệm nào làm ngưỡng bị lệch khi chuyển site?

Với mỗi thiết lập (seed validation 0), huấn luyện lại CBM (CBM3-Sign, lớp khái niệm v3, 2 seed) đúng như bước 5,
kiểm tra tái lập (so với điểm bước 5), rồi lưu:
  - dấu cổng (sign gate) và trọng số chuẩn hóa w_k = s_k · softplus(v_k) (trung bình 2 seed)
  - tham số mốc đã học
  - PHÂN RÃ dịch chuyển logit: với chuẩn hóa nguồn cố định, logit = Σ_k w_k z_k + b, nên chênh lệch trung bình logit
    của nhóm âm tính (và dương tính) giữa đích và validation nguồn = Σ_k w_k (z̄_k,đích − z̄_k,nguồn).
    Mỗi số hạng là phần đóng góp của khái niệm k vào việc ngưỡng nguồn bị lệch. Không dùng nhãn đích để chọn gì
    cả; nhãn đích chỉ dùng để tách lớp khi PHÂN TÍCH.
Chạy: python cbm_interp.py <harmonized> <concepts_v5> <out>
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
import cbm_lib4 as L  # noqa: E402
import concepts3  # noqa: E402

torch.set_num_threads(1)
HD, V5, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CH = int(sys.argv[4]) if len(sys.argv) > 4 else 0
NCH = int(sys.argv[5]) if len(sys.argv) > 5 else 1
X, Y = L.load(HD)
DOM = L.DOMAINS
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
NAMES = concepts3.CONCEPT_NAMES
V5ROWS = {(r["setting"], r["vseed"]): r for f in ["v5_raw_0.jsonl", "v5_raw_1.jsonl"]
          for r in map(json.loads, open(os.path.join(V5, f)))}
out = []
for i in list(range(9))[CH::NCH]:
    fn = os.path.join(OUT, f"cbm_interp_{i}.json")
    if os.path.exists(fn):
        continue
    src, tgt = SETTINGS[i]
    xtr, ytr, dtr, di = L.pack(X, Y, src)
    tr, va = L.strat_val(ytr, di, 0)
    signs = L.sign_gate(xtr[tr], ytr[tr], dtr[tr], cv="v3")
    ms = [L.train_cbm2(xtr[tr], ytr[tr], dtr[tr], xtr[va], ytr[va], dtr[va], s, False, signs, cv="v3") for s in (0, 1)]
    st = L.cbm_predict(ms, X[tgt], False)
    repro = float(np.abs(st - np.array(V5ROWS[(i, 0)]["models"]["CBM3-Sign"]["tgt"])).max())
    W, FID, dneg, dpos = [], [], [], []
    yv, yt = ytr[va], Y[tgt]
    for m in ms:
        with torch.no_grad():
            w = m.weights().numpy(); mu, sd = [t.numpy() for t in m.stats["pool"]]
            zv = (m.concept_values(xtr[va]) - mu) / sd
            zt = (m.concept_values(X[tgt]) - mu) / sd
        W.append(w)
        FID.append({k: float(v) for k, v in m.concepts.fiducial_values().items()} if hasattr(m.concepts, "fiducial_values") else {})
        dneg.append(w * (zt[yt == 0].mean(0) - zv[yv == 0].mean(0)))
        dpos.append(w * (zt[yt == 1].mean(0) - zv[yv == 1].mean(0)))
    W, dneg, dpos = np.mean(W, 0), np.mean(dneg, 0), np.mean(dpos, 0)
    rec = dict(setting=i, train="+".join(src), test=tgt, repro=repro, signs=signs.tolist(), weights=W.tolist(),
               fiducials=FID, shift_neg=dneg.tolist(), shift_pos=dpos.tolist(),
               shift_neg_total=float(dneg.sum()), shift_pos_total=float(dpos.sum()))
    json.dump(rec, open(fn, "w"))
    top = np.argsort(-np.abs(dneg))[:4]
    print(i, "+".join(src), "->", tgt, f"repro {repro:.1e}", "Δlogit neg/pos", round(dneg.sum(), 2), round(dpos.sum(), 2),
          "top neg:", [(NAMES[k], round(float(dneg[k]), 2)) for k in top], flush=True)
print("xong")
