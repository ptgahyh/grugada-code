"""(Bước 5) Vì sao QRS_sd15 bền với thiết bị mà KHÔNG giúp CBM? Thông tin riêng của QRS_w là gì?

  I1  phần thông tin riêng: hồi quy khái niệm QRS lên 27 khái niệm còn lại, AUROC của phần dư;
      tương quan Spearman lớn nhất với một khái niệm khác
  I2  hồi quy logistic trên khái niệm cố định (không học mốc), chéo miền:
      không có QRS / QRS_w / QRS_sd15 / cả hai
  I3  giả thuyết "QRS phân mảnh" (fragmented QRS): tỉ lệ năng lượng tần số cao (> 15 Hz) trong cửa sổ
      [−80, +150] ms của V1–V3; AUROC theo miền và tương quan với QRS_w

Chạy: python qrs_info5.py <harmonized> <output>
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import concepts5  # noqa: E402
from concepts5 import ANCHOR, concept_names  # noqa: E402

torch.set_num_threads(1)
HD, OUT = sys.argv[1], sys.argv[2]
N = concept_names("abs")
E = {w: concepts5.BrugadaConcepts(learn=False, qrs=w) for w in ["abs", "sd15"]}
D = {d: np.load(os.path.join(HD, f"{d}.npz")) for d in ["huca", "claris", "mortara"]}
C, HF, rep = {}, {}, {"I1": {}, "I2": {}, "I3": {}}
for d in D:
    x = torch.tensor(D[d]["X"].astype(np.float32)); y = D[d]["y"]
    with torch.no_grad():
        C[d] = {w: E[w](x).numpy() for w in E}
        v = x[:, 3:6]; hf = v - E["abs"].lowpass(v); w = slice(ANCHOR - 8, ANCHOR + 15)
        HF[d] = ((hf[:, :, w] ** 2).sum((1, 2)) / ((v[:, :, w] - v[:, :, w].mean(-1, keepdim=True)) ** 2).sum((1, 2))).numpy()
    C[d]["none"] = C[d]["abs"][:, :-1]
    C[d]["both"] = np.c_[C[d]["abs"], C[d]["sd15"][:, -1:]]
    rep["I1"][d] = {}
    for m in ["abs", "sd15"]:
        q = C[d][m][:, -1]; base = C[d]["none"]
        res = q - LinearRegression().fit(base, q).predict(base)
        rho = [abs(spearmanr(q, base[:, k])[0]) for k in range(27)]
        rep["I1"][d][m] = dict(auroc_alone=round(float(roc_auc_score(y, q)), 3), auroc_residual=round(float(roc_auc_score(y, res)), 3),
                               max_rho_other=round(float(max(rho)), 2), most_similar=N[int(np.argmax(rho))])
    q = C[d]["abs"][:, -1]; res = q - LinearRegression().fit(C[d]["none"], q).predict(C[d]["none"])
    rep["I3"][d] = dict(hf_auroc=round(float(roc_auc_score(y, HF[d])), 3), rho_qrsw_hf=round(float(spearmanr(q, HF[d])[0]), 2),
                        rho_qrsw_resid_hf=round(float(spearmanr(res, HF[d])[0]), 2),
                        hf_mean_neg=round(float(HF[d][y == 0].mean()), 3), hf_mean_pos=round(float(HF[d][y == 1].mean()), 3))
for tr in D:
    for te in D:
        if tr == te:
            continue
        rep["I2"][f"{tr}->{te}"] = {}
        for m in ["none", "abs", "sd15", "both"]:
            sc = StandardScaler().fit(C[tr][m])
            lr = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(C[tr][m]), D[tr]["y"])
            rep["I2"][f"{tr}->{te}"][m] = round(float(roc_auc_score(D[te]["y"], lr.predict_proba(sc.transform(C[te][m]))[:, 1])), 3)
json.dump(rep, open(os.path.join(OUT, "qrs_info5.json"), "w"), indent=1, ensure_ascii=False)
for k, v in rep.items():
    for kk, vv in v.items():
        print(k, kk, vv)
