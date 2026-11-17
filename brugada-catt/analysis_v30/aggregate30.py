"""Bước 30: gộp 5 lần huấn luyện. Xem PLAN_vi.md.

Chạy: python analysis_v30/aggregate30.py <thư mục v30 RBBB/LVH (rep0..rep4/{rbbb,lvh,lvhmv})> <thư mục Brugada (res_r1..res_r4)>
Brugada lần 0 = analysis_v27/results. Ghi analysis_v30/results/*.csv và analysis_v30/summary_v30.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

H = os.path.dirname(os.path.abspath(__file__))
V30, BRU = sys.argv[1], sys.argv[2]
OUT = os.path.join(H, "results"); os.makedirs(OUT, exist_ok=True)
T975 = None  # tính theo số lần thực có
COMPS = ("SRC", "GMM-2", "Otsu")
PRIMARY = {"rbbb": "CATT(A2)", "lvh": "CATT(A3)", "lvhmv": "CATT(A3)"}
LVHMV_BROKEN0 = ["cn→ptb_cs", "cn→ptb_at"]


def pooled_rbbb_lvh(d, cat, m, sets):
    """Như phần P4 của analyze_rbbb/analyze_lvh: trung bình qua thiết lập, bootstrap theo thứ tự hàng trong từng thiết lập."""
    pt = pd.read_csv(os.path.join(d, "p4_point.csv"), index_col=0); bt = pd.read_csv(os.path.join(d, "p4_boot.csv"))
    sets = list(pt.index) if sets is None else sets
    v = np.mean([bt[bt.setting == s][cat].to_numpy() - bt[bt.setting == s][m].to_numpy() for s in sets], 0)
    p0 = float(np.mean([pt.loc[s, cat] - pt.loc[s, m] for s in sets]))
    lo, hi = np.percentile(v, [2.5, 97.5])
    broken = [s for s in pt.index if pt.loc[s, "SRC"] < pt.loc[s, "ORACLE"] - 0.05]
    return p0, float(lo), float(hi), broken


def brugada(d, m, split):
    if split == "prev5":
        t = pd.read_csv(os.path.join(d, "v8_prev5_pooled_diff.csv")); r = t[t.vs == m].iloc[0]
    else:
        t = pd.read_csv(os.path.join(d, "v8_bootstrap_pooled.csv")); r = t[(t.vs == m) & (t.split == split)].iloc[0]
    ps = pd.read_csv(os.path.join(d, "v8_bootstrap_per_setting.csv"))
    w = ps.pivot(index="setting", columns="vs", values="delta")          # CATT − m theo thiết lập
    broken = [int(s) for s in w.index if (w.loc[s, "ORACLE"] - w.loc[s, "SRC"]) < -0.05]   # SRC − ORACLE < −0,05
    return float(r.delta), float(r.lo), float(r.hi), broken


rows, broken_rows = [], []
for prob in ("rbbb", "lvh", "lvhmv"):
    reps = sorted(int(x[3:]) for x in os.listdir(V30) if x.startswith("rep") and os.path.exists(os.path.join(V30, x, prob, "p4_boot.csv")))
    scopes = [("tất cả 6", None)] if prob != "lvhmv" else [("hỏng (lần 0)", LVHMV_BROKEN0), ("không hỏng (lần 0)", "rest"), ("tất cả 6", None)]
    for scope, sets in scopes:
        for m in COMPS:
            for r in reps:
                d = os.path.join(V30, f"rep{r}", prob)
                if sets == "rest":
                    idx = list(pd.read_csv(os.path.join(d, "p4_point.csv"), index_col=0).index); ss = [s for s in idx if s not in LVHMV_BROKEN0]
                else:
                    ss = sets
                p0, lo, hi, br = pooled_rbbb_lvh(d, PRIMARY[prob], m, ss)
                rows.append(dict(problem=prob, catt=PRIMARY[prob], scope=scope, vs=m, rep=r, delta=p0, lo=lo, hi=hi))
                if scope.startswith("tất cả") and m == "SRC":
                    broken_rows.append(dict(problem=prob, rep=r, n_broken=len(br), broken=";".join(br)))
b_reps = [(0, os.path.join(H, "..", "analysis_v27", "results"))] + \
         [(r, os.path.join(BRU, f"res_r{r}")) for r in (1, 2, 3, 4) if os.path.exists(os.path.join(BRU, f"res_r{r}", "v8_prev5_pooled_diff.csv"))]
for split, scope in (("held-out", "held-out"), ("prev5", "tỉ lệ 5%")):
    for m in COMPS:
        for r, d in b_reps:
            p0, lo, hi, br = brugada(d, m, split)
            rows.append(dict(problem="brugada", catt="CATT", scope=scope, vs=m, rep=r, delta=p0, lo=lo, hi=hi))
            if split == "held-out" and m == "SRC":
                broken_rows.append(dict(problem="brugada", rep=r, n_broken=len(br), broken=";".join(map(str, br))))
R = pd.DataFrame(rows); R.round(5).to_csv(os.path.join(OUT, "per_rep.csv"), index=False)
pd.DataFrame(broken_rows).to_csv(os.path.join(OUT, "broken_per_rep.csv"), index=False)

summ = []
for (prob, scope, m), g in R.groupby(["problem", "scope", "vs"], sort=False):
    k = len(g); mean = g.delta.mean(); s = g.delta.std(ddof=1) if k > 1 else np.nan
    h = ((g.hi - g.lo) / 2).median(); t = stats.t.ppf(0.975, k - 1) if k > 1 else np.nan
    lo_t, hi_t = mean - t * s / np.sqrt(k), mean + t * s / np.sqrt(k)
    same = int((np.sign(g.delta) == np.sign(mean)).sum())
    verdict = ("không phân biệt được" if (k < 2 or lo_t <= 0 <= hi_t or same < k - 1)
               else ("CATT tốt hơn" if mean > 0 else "CATT kém hơn"))
    summ.append(dict(problem=prob, scope=scope, vs=m, n_reps=k, deltas=" ".join(f"{x:+.4f}" for x in g.delta), mean=mean, sd_between=s,
                     boot_halfwidth_median=h, ratio_sd_to_bootSE=s / (h / 1.96) if h > 0 else np.nan,
                     same_sign=f"{same}/{k}", t_lo=lo_t, t_hi=hi_t, V2=verdict))
S = pd.DataFrame(summ); S.round(4).to_csv(os.path.join(OUT, "summary_table.csv"), index=False)
out = {"V1_bootstrap_understates": {f"{r.problem} / {r.scope}": bool(r.ratio_sd_to_bootSE > 1)
                                    for r in S[S.vs == "SRC"].itertuples()},
       "V2": {f"{r.problem} / {r.scope} / CATT − {r.vs}": r.V2 for r in S.itertuples()},
       "V3_brugada_main_holds": bool(((S.problem == "brugada") & (S.scope == "held-out") & (S.vs == "SRC") & (S.V2 == "CATT tốt hơn")).any()),
       "V4_broken_counts": {p: g.n_broken.tolist() for p, g in pd.DataFrame(broken_rows).groupby("problem")}}
json.dump(out, open(os.path.join(H, "summary_v30.json"), "w"), indent=1, ensure_ascii=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(S.round(4).to_string(index=False)); print(json.dumps(out, indent=1, ensure_ascii=False))
