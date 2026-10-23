"""1-, 2- and 3-lead CNNs with the full protocol (9 settings x 3 validation seeds, 3-seed ensemble).

Re-runs analysis_v8/lead_subsets.py (one validation seed) and saves per-patient scores for the bootstrap.
The V1-V3 model is the paper's CNN-V1V3 (same code and seeds as concepts_v5/compare_v5.py): it is re-trained
on settings 0-1 as a reproducibility check, and its stored scores are reused on settings 2-8.

Run: python analysis_v10/leads_3seed.py harmonized concepts_v5 analysis_v10
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HELD, SETTINGS, VSEEDS, ScoreStore, cbm_pseudo, cnn_predict, load, metrics, pack,  # noqa: E402
                    strat_val, stv2, train_cnn, v5_raw, youden, bacc)

HD, V5, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(1)
X, Y = load(HD)
R5 = v5_raw(V5)
SUBSETS = {"V1": [3], "V2": [4], "V3": [5], "V1+V2": [3, 4], "V1-V3": [3, 4, 5], "I": [0], "II": [1],
           "9 leads": list(range(9))}
store = ScoreStore(os.path.join(OUT, "leads_3seed_scores.npz"))
t0 = time.time()
for k, (src, tgt) in enumerate(SETTINGS):
    xtr, ytr, dtr, di = pack(X, Y, src)
    for vs in VSEEDS:
        tr, va = strat_val(ytr, di, vs)
        assert ytr[va].tolist() == R5[(k, vs)]["y_val"]
        if k >= 2 and not store.has("V1-V3", k, vs):      # identical to the paper's CNN-V1V3: reuse its scores
            r5m = R5[(k, vs)]["models"]["CNN-V1V3"]; store.put("V1-V3", k, vs, r5m["val"], r5m["tgt"])
        for name, idx in SUBSETS.items():
            if store.has(name, k, vs):
                continue
            a = xtr[:, idx]
            nets = [train_cnn(a[tr], ytr[tr], a[va], ytr[va], s) for s in (0, 1, 2)]
            store.put(name, k, vs, cnn_predict(nets, a[va]), cnn_predict(nets, X[tgt][:, idx]))
        store.save()
        print(k, "+".join(src), "->", tgt, "vseed", vs, f"{time.time() - t0:.0f}s", flush=True)

# ---- consistency check against the paper's CNN-V1V3 scores
dev = max(np.abs(store.get("V1-V3", k, vs)[1] - np.array(R5[(k, vs)]["models"]["CNN-V1V3"]["tgt"])).max()
          for k in (0, 1) for vs in VSEEDS)
print(f"re-trained V1-V3 on settings 0-1 vs paper CNN-V1V3: max |score difference| = {dev:.2e} (settings 2-8 reuse the paper scores)")

# ---- metrics: AUROC on all 9 settings; thresholds on 7 held-out settings
rows = []
for name in SUBSETS:
    for k, (src, tgt) in enumerate(SETTINGS):
        xtr, ytr, dtr, di = pack(X, Y, src)
        for vs in VSEEDS:
            tr, va = strat_val(ytr, di, vs)
            sv, st = store.get(name, k, vs)
            m = metrics(ytr[va], sv, Y[tgt], st, cbm_pseudo(R5[(k, vs)]))
            pl_st = (stv2(X[tgt]) >= youden(ytr[va], stv2(xtr[va]))).astype(int)   # single-lead (V2) labeller
            m["CATT_STV2"] = bacc(Y[tgt], st, youden(pl_st, st))
            rows.append(dict(input=name, k=k, train="+".join(src), test=tgt, vseed=vs, **m))
df = pd.DataFrame(rows)
df.round(4).to_csv(os.path.join(OUT, "leads_3seed_raw.csv"), index=False)
per = df.groupby(["input", "k"])[["auroc", "SRC", "CATT", "CATT_STV2", "ORACLE"]].mean().reset_index()
A = per.groupby("input")["auroc"].agg(["mean", "min"]).add_prefix("auroc9_")
H = per[per.k.isin(HELD)].groupby("input")[["auroc", "SRC", "CATT", "CATT_STV2", "ORACLE"]].agg(["mean", "min"])
H.columns = [f"held_{a}_{b}" for a, b in H.columns]
summ = A.join(H).reindex(list(SUBSETS)).round(3)
summ.to_csv(os.path.join(OUT, "leads_3seed_summary.csv"))
print(summ.to_string())
