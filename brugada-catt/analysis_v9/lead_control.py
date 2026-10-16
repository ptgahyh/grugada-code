"""(Bước 9) Đối chứng: V1–V3 tốt hơn vì là chuyển đạo Brugada, hay chỉ vì ít đầu vào hơn (ít overfit)?

Huấn luyện CNN với 3 chuyển đạo NGẪU NHIÊN (cùng kiến trúc, tăng cường, dừng sớm như CNN-V1V3) trên 9 thiết lập ngoài
miền, seed validation 0, ensemble 3 seed (0, 1, 2) — đúng giao thức của CNN-V1V3 trong compare_v5.py, nên so trực tiếp
được với điểm CNN-V1V3 (vseed 0) trong v5_raw. Thêm CNN 9 chuyển đạo cùng split để làm mốc.

Bộ 3 chuyển đạo: 20 bộ rút ngẫu nhiên (seed 0) trong 83 bộ khác V1–V3, cộng bộ (I, II, V5) mà phản biện nêu.
Thứ tự chuyển đạo trong dữ liệu: I, II, III, V1, V2, V3, V4, V5, V6 (chỉ số 0..8).

Chạy: python lead_control.py <harmonized> <output> <chunk> <số chunk>   ->  lead_control_<chunk>.jsonl
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "concepts_v4"))
import cbm_lib4 as L  # noqa: E402

torch.set_num_threads(1)
HD, OUT, CH, NCH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
NAMES = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
ALL = [c for c in itertools.combinations(range(9), 3) if c != (3, 4, 5)]
rng = np.random.default_rng(0)
TRIPLETS = [ALL[k] for k in rng.choice(len(ALL), 20, replace=False)]
if (0, 1, 7) not in TRIPLETS:
    TRIPLETS.append((0, 1, 7))
VARIANTS = [("9 leads", tuple(range(9)))] + [("+".join(NAMES[j] for j in t), t) for t in TRIPLETS]

X, Y = L.load(HD)
DOM = L.DOMAINS
SETTINGS = [([a], b) for a in DOM for b in DOM if a != b] + [([a for a in DOM if a != b], b) for b in DOM]
JOBS = [(v, i) for v in range(len(VARIANTS)) for i in range(len(SETTINGS))][CH::NCH]
os.makedirs(OUT, exist_ok=True)
fout = os.path.join(OUT, f"lead_control_{CH}.jsonl")
done = set()
if os.path.exists(fout):
    done = {(r["variant"], r["setting"]) for r in map(json.loads, open(fout))}
t0 = time.time()
for v, i in JOBS:
    name, leads = VARIANTS[v]
    if (name, i) in done:
        continue
    src, tgt = SETTINGS[i]
    xtr, ytr, dtr, di = L.pack(X, Y, src)
    tr, va = L.strat_val(ytr, di, 0)
    li = list(leads)
    nets = [L.train_cnn(xtr[tr][:, li], ytr[tr], xtr[va][:, li], ytr[va], s) for s in (0, 1, 2)]
    st = L.cnn_predict(nets, X[tgt][:, li])
    rec = dict(variant=name, leads=li, setting=i, train="+".join(src), test=tgt,
               has_right_precordial=bool(set(li) & {3, 4, 5}), auroc=float(roc_auc_score(Y[tgt], st)))
    with open(fout, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(name, i, "+".join(src), "->", tgt, round(rec["auroc"], 3), f"{time.time() - t0:.0f}s", flush=True)
print("xong", f"{time.time() - t0:.0f}s")
