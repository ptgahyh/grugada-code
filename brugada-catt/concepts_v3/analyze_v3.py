"""Phân tích bước 3: so sánh công bằng (v3_raw.json), stress test Ada (stress_raw.json),
chẩn đoán Mortara (diag_mortara.json). Xuất bảng + 3 hình.

Chạy: python analyze_v3.py <thư mục concepts_v3> <harmonized>
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OD, HD = sys.argv[1], sys.argv[2]
Y = {n: np.load(os.path.join(HD, f"{n}.npz"))["y"].astype(int) for n in ["huca", "claris", "mortara"]}
LAB = {"huca": "HUCA", "claris": "Claris", "mortara": "Mortara"}
ORDER = [("huca", "claris"), ("huca", "mortara"), ("claris", "huca"), ("claris", "mortara"), ("mortara", "huca"),
         ("mortara", "claris"), ("claris+mortara", "huca"), ("huca+mortara", "claris"), ("huca+claris", "mortara")]
name = lambda tr, te: " + ".join(LAB[s] for s in tr.split("+")) + " → " + LAB[te]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#9a9993", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb"})
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def fast_auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_delta(y, s1, s2, n=2000, seed=7):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(fast_auc(y[i], s1[i]) - fast_auc(y[i], s2[i]))
    d = np.array(d)
    return fast_auc(y, s1) - fast_auc(y, s2), np.percentile(d, 2.5), np.percentile(d, 97.5), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ 1. so sánh công bằng
raw = pd.DataFrame(json.load(open(os.path.join(OD, "v3_raw.json"))))
S, rows = {}, []
for (tr, te, m), g in raw.groupby(["train", "test", "model"]):
    s = np.mean([np.array(v) for v in g.scores], 0); S[(tr, te, m)] = s
    rows.append(dict(train=tr, test=te, model=m, auroc=round(fast_auc(Y[te], s), 3), seed_sd=round(g.auroc.std(ddof=0), 3),
                     sens=round(g.sens.mean(), 3), spec=round(g.spec.mean(), 3), bal_acc=round(((g.sens + g.spec) / 2).mean(), 3)))
res = pd.DataFrame(rows)
comps = []
for m in ["CBM2-SignAda", "Hybrid", "CBM2-SignAda-noShift", "CNN"]:
    for ref in ["CNN-V1V3", "CNN"]:
        if m == ref:
            continue
        block = []
        for tr, te in ORDER:
            if (tr, te, m) in S and (tr, te, ref) in S:
                d, lo, hi, p = boot_delta(Y[te], S[(tr, te, m)], S[(tr, te, ref)])
                block.append(dict(model=m, ref=ref, train=tr, test=te, delta=round(d, 3), lo=round(lo, 3), hi=round(hi, 3), p=p))
        for b, a in zip(block, holm([b["p"] for b in block])):
            b["p_holm"] = round(float(a), 4)
            b["verdict"] = "thắng" if (a < 0.05 and b["delta"] > 0) else ("thua" if (a < 0.05 and b["delta"] < 0) else "hòa")
        comps += block
comp = pd.DataFrame(comps)
summ = res.groupby("model").agg(mean_auroc=("auroc", "mean"), mean_bal_acc=("bal_acc", "mean"), mean_sens=("sens", "mean"),
                                mean_spec=("spec", "mean"), seed_sd=("seed_sd", "mean"), n=("auroc", "size")).round(3)
wl = comp.groupby(["model", "ref"]).verdict.value_counts().unstack(fill_value=0)
pd.set_option("display.width", 250)
print(res.pivot_table(index=["train", "test"], columns="model", values="auroc").to_string())
print(res.pivot_table(index=["train", "test"], columns="model", values="bal_acc").to_string())
print(summ.to_string()); print(wl.to_string())
res.to_csv(os.path.join(OD, "v3_results.csv"), index=False); comp.to_csv(os.path.join(OD, "v3_paired_holm.csv"), index=False)
summ.to_csv(os.path.join(OD, "v3_summary.csv"))

# hình 1: AUROC và balanced accuracy theo thiết lập
models = [("CNN", C2, "CNN 9 chuyển đạo"), ("CNN-V1V3", C4, "CNN V1–V3"), ("CBM2-SignAda", C1, "CBM2-SignAda"), ("Hybrid", C3, "Hybrid")]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
yy = np.arange(len(ORDER))
for k, (m, c, lb) in enumerate(models):
    r = res.set_index(["train", "test", "model"])
    for j, col in enumerate(["auroc", "bal_acc"]):
        v = [r.loc[(tr, te, m), col] for tr, te in ORDER]
        ax[j].plot(v, yy + (k - 1.5) * 0.17, "o", color=c, ms=6, label=lb)
for j, (t, xl) in enumerate([("Phân biệt", "AUROC (điểm trung bình 3 seed)"), ("Ngưỡng ở miền mới", "Balanced accuracy (ngưỡng chọn trên train)")]):
    ax[j].set_title(t, loc="left", fontsize=11, color="#0b0b0b"); ax[j].set_xlabel(xl)
    ax[j].grid(axis="x", color="#e4e3df"); ax[j].set_axisbelow(True)
    for s in ["top", "right"]:
        ax[j].spines[s].set_visible(False)
ax[0].set_yticks(yy); ax[0].set_yticklabels([name(a, b) for a, b in ORDER]); ax[0].invert_yaxis()
fig.legend(*ax[1].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=4, fontsize=9)
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(os.path.join(OD, "fig_v3_compare.png"), dpi=150); plt.close()

# ------------------------------------------------------------------ 2. stress test Ada
st = pd.DataFrame(json.load(open(os.path.join(OD, "stress_raw.json"))))
prev = st[st.kind == "prev"].groupby(["value", "method"])[["auroc", "sens", "spec", "bal_acc"]].mean().reset_index()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for m, c, lb in [("Sign", C2, "Sign (không thích nghi)"), ("SignAda", C1, "SignAda"), ("SignAda-robust", C3, "SignAda (trung vị/IQR)")]:
    d = prev[prev.method == m]
    ax[0].plot(d.value * 100, d.bal_acc, "-o", color=c, lw=2, ms=6, label=lb)
    ax[1].plot(d.value * 100, d.sens, "-o", color=c, lw=2, ms=6)
    ax[1].plot(d.value * 100, d.spec, "--s", color=c, lw=2, ms=5)
ax[0].set_title("Balanced accuracy theo tỉ lệ Brugada trong lô", loc="left", fontsize=11, color="#0b0b0b")
ax[1].set_title("Độ nhạy (liền) và độ đặc hiệu (nét đứt)", loc="left", fontsize=11, color="#0b0b0b")
for a in ax:
    a.set_xlabel("Tỉ lệ Brugada trong lô 100 bệnh nhân (%)"); a.grid(color="#e4e3df"); a.set_axisbelow(True)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
ax[0].legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OD, "fig_v3_prevalence.png"), dpi=150); plt.close()
print(prev.round(3).to_string(index=False))

# ------------------------------------------------------------------ 3. CNN bị giới hạn thông tin
dg = json.load(open(os.path.join(OD, "diag_mortara.json")))
ab = pd.DataFrame(dg["cnn_restriction"]).set_index(["train", "test"]).loc[ORDER].reset_index()
fig, ax = plt.subplots(figsize=(7.5, 5.6))
for k, (col, c, lb) in enumerate([("CNN", C2, "9 chuyển đạo"), ("CNN-V1V3", C4, "chỉ V1–V3"), ("CNN-noST", C1, "9 chuyển đạo, xóa ST-T"), ("CNN-QRS", C3, "9 chuyển đạo, chỉ QRS")]):
    ax.plot(ab[col], yy + (k - 1.5) * 0.17, "o", color=c, ms=6, label=lb)
ax.set_yticks(yy); ax.set_yticklabels([name(a, b) for a, b in ORDER]); ax.invert_yaxis()
ax.set_xlabel("AUROC ngoài miền (seed validation 0)"); ax.grid(axis="x", color="#e4e3df"); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2)
ax.set_title("CNN học gì? Giới hạn thông tin đầu vào", loc="left", fontsize=11, color="#0b0b0b")
plt.tight_layout(); plt.savefig(os.path.join(OD, "fig_v3_cnn_restriction.png"), dpi=150); plt.close()
print(ab.mean(numeric_only=True).round(3).to_dict())
