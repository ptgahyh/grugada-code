"""Dịch chuyển khái niệm giữa các miền ở nhóm âm tính (SMD), so sánh điểm J từ 9 chuyển đạo (v2)
và chỉ từ V1–V3 (v3). Tham số mốc khởi tạo, không học.

Chạy: python shift_v4.py <harmonized> <output>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cbm_lib4 as L  # noqa: E402
from concepts2 import CONCEPT_NAMES as N  # noqa: E402

X, Y = L.load(sys.argv[1])
PAIRS = [("claris", "mortara"), ("huca", "claris"), ("huca", "mortara")]
res = {}
for cv in ["v2", "v3"]:
    C = {d: L.concepts_fixed(X[d], cv) for d in L.DOMAINS}
    res[cv] = {}
    for a, b in PAIRS:
        x, y = C[a][Y[a] == 0], C[b][Y[b] == 0]
        smd = (y.mean(0) - x.mean(0)) / np.sqrt((x.var(0) + y.var(0)) / 2)
        res[cv][f"{a}->{b}"] = dict(QRS_w=round(float(smd[-1]), 3), median_abs_others=round(float(np.median(np.abs(smd[:-1]))), 3),
                                    per_concept={n: round(float(v), 3) for n, v in zip(N, smd)})
        print(cv, f"{a}->{b}", "QRS_w", res[cv][f"{a}->{b}"]["QRS_w"], "|SMD| trung vị các khái niệm khác",
              res[cv][f"{a}->{b}"]["median_abs_others"])
json.dump(res, open(os.path.join(sys.argv[2], "shift_v4.json"), "w"), indent=1)
