"""Bước 15c: kiểm F1 (ước lượng SD(t̂) không nhãn bằng mẫu con) trên Brugada. Code cố định từ commit kế hoạch bước 15.

Brugada không được dùng khi phát triển F1 (chỉ dùng RBBB, bước 15a), nên đây là một phép kiểm ngoài mẫu thứ hai.
Với mỗi dòng của analysis_v13/brugada_unified_v2.csv (thiết lập, seed, mỏ neo, n), lấy 20 mẫu cỡ n không nhãn và tính
SD dự đoán bằng common15.sd_subsample (B = 100, b = n/2). So với sd_obs và sd_plugin (KDE) đã có trong tệp đó.
Tiêu chí như F1 trong PLAN_vi.md: trung vị dự đoán/quan sát ∈ [0,8; 1,25] và Spearman ≥ 0,7.

Chạy (máy có analysis_v11/scores):
  python analysis_v15/brugada_subsample_check.py analysis_v11 analysis_v13/brugada_unified_v2.csv analysis_v15
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common15 import logit, sd_from_c, sd_subsample, youden  # noqa: E402

V11, V2, OUT = sys.argv[1:4]
LABS = {"CBM-sd15": "CBM5", "CBM": "CBM3", "ST_V2": "ST_V2", "LogReg9": "LogReg9", "CNN9": "CNN9",
        "ResNet": "ResNet", "CNNalt": "CNNalt"}
rng = np.random.default_rng(1503)
D = pd.read_csv(V2)
sub = []
for (i, vs), d in D.groupby(["setting", "vseed"]):
    Z = np.load(os.path.join(V11, "scores", f"lab_{i}_{vs}.npz")); f = logit(Z["f_tgt"]); N = len(f)
    for r in d.itertuples():
        k = LABS[r.anchor]
        pl = (Z[f"{k}_tgt"] >= youden(Z["y_val"], Z[f"{k}_val"])).astype(int)
        preds = []
        for _ in range(20):
            idx = rng.choice(N, int(r.n), replace=False)
            c = sd_subsample(f[idx], pl[idx], rng)
            if c is not None:
                preds.append(sd_from_c(c, r.n, N))
        sub.append(float(np.median(preds)) if preds else np.nan)
    print("setting", i, "seed", vs, flush=True)
D["sd_sub"] = sub
D[["setting", "vseed", "anchor", "N", "n", "sd_obs", "sd_plugin", "sd_pred_1.0", "sd_sub"]].round(6).to_csv(
    os.path.join(OUT, "brugada_subsample_check.csv"), index=False)


def stats(d, col):
    d = d.dropna(subset=[col]); r = np.log(d[col] / d.sd_obs)
    return dict(median_pred_over_obs=round(float(np.exp(r.median())), 3),
                iqr=[round(float(np.exp(r.quantile(q))), 3) for q in (.25, .75)],
                spearman=round(float(spearmanr(d[col], d.sd_obs)[0]), 3), median_abs_log_err=round(float(r.abs().median()), 3),
                n=len(d))


S = {c: stats(D, c) for c in ("sd_pred_1.0", "sd_plugin", "sd_sub")}
S["sd_sub"]["criterion_met"] = bool(0.8 <= S["sd_sub"]["median_pred_over_obs"] <= 1.25 and S["sd_sub"]["spearman"] >= 0.7)
S["by_anchor"] = {a: {c: stats(g, c)["median_pred_over_obs"] for c in ("sd_pred_1.0", "sd_plugin", "sd_sub")}
                  for a, g in D.groupby("anchor")}
json.dump(S, open(os.path.join(OUT, "brugada_subsample_check.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(S, indent=1, ensure_ascii=False))
