"""Phân tích kết quả v2: AUROC (điểm trung bình 3 seed validation), độ lệch giữa seed,
balanced accuracy, so sánh cặp bootstrap với hiệu chỉnh Holm trên 9 thiết lập ngoài miền.

Chạy: python analyze_v2.py <thư mục concepts_v2> <thư mục concepts (v1)> <thư mục harmonized>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OD, V1, HD = sys.argv[1], sys.argv[2], sys.argv[3]
raw = json.load(open(os.path.join(OD, "v2_raw.json")))
Y = {n: np.load(os.path.join(HD, f"{n}.npz"))["y"].astype(int) for n in ["huca", "claris", "mortara"]}

# v1 CBM (1 seed validation) để tham chiếu
v1 = np.load(os.path.join(V1, "concept_scores.npy"), allow_pickle=True).item()

rows, S = [], {}
df = pd.DataFrame(raw)
for (tr, te, m), g in df.groupby(["train", "test", "model"], sort=False):
    s = np.mean([np.array(v) for v in g.scores], 0)
    S[(tr, te, m)] = s
    rows.append(dict(train=tr, test=te, model=m, auroc_avgscore=round(roc_auc_score(Y[te], s), 3),
                     auroc_seed_mean=round(g.auroc.mean(), 3), auroc_seed_sd=round(g.auroc.std(ddof=0), 3),
                     sens=round(g.sens.mean(), 3), spec=round(g.spec.mean(), 3),
                     bal_acc=round(((g.sens + g.spec) / 2).mean(), 3)))
for (tr, te) in {(r["train"], r["test"]) for r in rows}:
    proto = "lodo" if "+" in tr else "cross"
    key = f"{proto}|{tr}|{te}|CBM"
    if key in v1:
        y, s = v1[key]
        S[(tr, te, "CBM1")] = s
        rows.append(dict(train=tr, test=te, model="CBM1", auroc_avgscore=round(roc_auc_score(y, s), 3),
                         auroc_seed_mean=np.nan, auroc_seed_sd=np.nan, sens=np.nan, spec=np.nan, bal_acc=np.nan))
res = pd.DataFrame(rows)


def fast_auc(y, s):
    """AUROC qua hạng (Mann-Whitney), xử lý điểm trùng bằng hạng trung bình."""
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_delta(y, s1, s2, n=2000, seed=7):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(fast_auc(y[i], s1[i]) - fast_auc(y[i], s2[i]))
    d = np.array(d)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return roc_auc_score(y, s1) - roc_auc_score(y, s2), np.percentile(d, 2.5), np.percentile(d, 97.5), p


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


comps = []
models = [m for m in res.model.unique() if m.startswith("CBM")]
settings = sorted({(r["train"], r["test"]) for r in rows}, key=lambda t: ("+" in t[0], t))
for m in models:
    for ref in ["CNN", "ST_V2"]:
        block = []
        for tr, te in settings:
            if (tr, te, m) in S and (tr, te, ref) in S:
                d, lo, hi, p = boot_delta(Y[te], S[(tr, te, m)], S[(tr, te, ref)])
                block.append(dict(model=m, ref=ref, train=tr, test=te, delta=round(d, 3), lo=round(lo, 3), hi=round(hi, 3), p=p))
        if block:
            adj = holm([b["p"] for b in block])
            for b, a in zip(block, adj):
                b["p_holm"] = round(float(a), 4)
                b["verdict"] = "thắng" if (a < 0.05 and b["delta"] > 0) else ("thua" if (a < 0.05 and b["delta"] < 0) else "hòa")
            comps += block
comp = pd.DataFrame(comps)

summary = res.groupby("model").agg(mean_auroc=("auroc_avgscore", "mean"), mean_seed_sd=("auroc_seed_sd", "mean"),
                                   mean_bal_acc=("bal_acc", "mean"), n_settings=("auroc_avgscore", "size")).round(3)
wl = comp.groupby(["model", "ref"]).verdict.value_counts().unstack(fill_value=0)
lodo = res[res.train.str.contains(r"\+")].groupby("model").auroc_avgscore.mean().round(3).rename("mean_auroc_lodo")
summary = summary.join(lodo)

pd.set_option("display.width", 250)
print(res.pivot_table(index=["train", "test"], columns="model", values="auroc_avgscore").round(3).to_string())
print(res.pivot_table(index=["train", "test"], columns="model", values="bal_acc").round(3).to_string())
print(summary.to_string()); print(wl.to_string())

# khái niệm bị loại bởi cổng dấu (LODO)
gate = {}
for r in raw:
    if r["model"] == "CBM2-Sign" and "+" in r["train"] and r["vseed"] == 0:
        s = r["extra"]["signs"]
        from_names = json.load(open(os.path.join(OD, "concept_names.json")))
        gate[r["train"] + "->" + r["test"]] = dict(
            dropped=[n for n, v in zip(from_names, s) if v == 0],
            negative=[n for n, v in zip(from_names, s) if v < 0])
print(json.dumps(gate, indent=1))

res.to_csv(os.path.join(OD, "v2_results.csv"), index=False)
comp.to_csv(os.path.join(OD, "v2_paired_holm.csv"), index=False)
summary.to_csv(os.path.join(OD, "v2_summary.csv"))
json.dump(gate, open(os.path.join(OD, "v2_sign_gate.json"), "w"), indent=1, ensure_ascii=False)
