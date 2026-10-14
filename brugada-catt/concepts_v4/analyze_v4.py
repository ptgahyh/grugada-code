"""Phân tích bước 4. Gộp v4_raw.json với v3_raw.json (CNN 9 chuyển đạo, ST_V2, CBM2-SignAda, Hybrid A2),
tính AUROC / balanced accuracy, so sánh cặp có hiệu chỉnh Holm, bảng kiến trúc, 2 hình.

Chạy: python analyze_v4.py <concepts_v4> <concepts_v3> <harmonized>
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

OD, V3, HD = sys.argv[1], sys.argv[2], sys.argv[3]
Y = {n: np.load(os.path.join(HD, f"{n}.npz"))["y"].astype(int) for n in ["huca", "claris", "mortara"]}
ORDER = [("huca", "claris"), ("huca", "mortara"), ("claris", "huca"), ("claris", "mortara"), ("mortara", "huca"),
         ("mortara", "claris"), ("claris+mortara", "huca"), ("huca+mortara", "claris"), ("huca+claris", "mortara")]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#9a9993", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb"})


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


v4 = pd.DataFrame(json.load(open(os.path.join(OD, "v4_raw.json"))))
v3 = pd.DataFrame(json.load(open(os.path.join(V3, "v3_raw.json"))))
v3 = v3[v3.model.isin(["CNN", "ST_V2", "CBM2-SignAda", "Hybrid"])].replace({"model": {"Hybrid": "Hybrid-A2"}})
raw = pd.concat([v4, v3], ignore_index=True)
S, rows = {}, []
for (tr, te, m), g in raw.groupby(["train", "test", "model"]):
    s = np.mean([np.array(v) for v in g.scores], 0); S[(tr, te, m)] = s
    rows.append(dict(train=tr, test=te, model=m, auroc=round(fast_auc(Y[te], s), 3), seed_sd=round(g.auroc.std(ddof=0), 3),
                     sens=round(g.sens.mean(), 3), spec=round(g.spec.mean(), 3), bal_acc=round(((g.sens + g.spec) / 2).mean(), 3),
                     worst_ba=round(((g.sens + g.spec) / 2).min(), 3)))
res = pd.DataFrame(rows)
summ = res.groupby("model").agg(mean_auroc=("auroc", "mean"), mean_bal_acc=("bal_acc", "mean"), min_bal_acc=("bal_acc", "min"),
                                mean_sens=("sens", "mean"), mean_spec=("spec", "mean"), seed_sd=("seed_sd", "mean")).round(3)
comps = []
for m in ["Hybrid-S3", "Hybrid-S2", "Hybrid-A3", "Hybrid-A2", "CBM3-Sign", "CBM3-SignAda", "CBM2-Sign"]:
    for ref in ["CNN-V1V3", "CNN", "CBM2-Sign"]:
        if m == ref:
            continue
        block = []
        for tr, te in ORDER:
            d, lo, hi, p = boot_delta(Y[te], S[(tr, te, m)], S[(tr, te, ref)])
            block.append(dict(model=m, ref=ref, train=tr, test=te, delta=round(d, 3), lo=round(lo, 3), hi=round(hi, 3), p=p))
        for b, a in zip(block, holm([b["p"] for b in block])):
            b["p_holm"] = round(float(a), 4)
            b["verdict"] = "thắng" if (a < 0.05 and b["delta"] > 0) else ("thua" if (a < 0.05 and b["delta"] < 0) else "hòa")
        comps += block
comp = pd.DataFrame(comps)
wl = comp.groupby(["model", "ref"]).verdict.value_counts().unstack(fill_value=0)
pd.set_option("display.width", 250)
print(res.pivot_table(index=["train", "test"], columns="model", values="auroc").to_string())
print(res.pivot_table(index=["train", "test"], columns="model", values="bal_acc").to_string())
print(summ.sort_values("mean_auroc").to_string()); print(wl.to_string())
res.to_csv(os.path.join(OD, "v4_results.csv"), index=False); comp.to_csv(os.path.join(OD, "v4_paired_holm.csv"), index=False)
summ.to_csv(os.path.join(OD, "v4_summary.csv"))

# hình: đánh đổi AUROC – balanced accuracy (trung bình 9 thiết lập)
# (màu, nhãn, ký hiệu, độ lệch nhãn); hình tròn = điểm J từ 9 chuyển đạo (v2), hình thoi = điểm J từ V1–V3 (v3)
show = {"ST_V2": ("#9a9993", "ST_V2 (1 đặc trưng)", "o", (8, 4)), "CNN": ("#eb6834", "CNN 9 chuyển đạo", "o", (8, 4)),
        "CNN-V1V3": ("#6250d6", "CNN V1–V3", "o", (8, -12)),
        "CBM2-Sign": ("#2a78d6", "CBM2-Sign", "o", (-62, -12)), "CBM2-SignAda": ("#2a78d6", "CBM2-SignAda", "o", (-40, 8)),
        "CBM3-Sign": ("#2a78d6", "CBM3-Sign", "D", (8, -12)), "CBM3-SignAda": ("#2a78d6", "CBM3-SignAda", "D", (8, 4)),
        "Hybrid-S2": ("#1baf7a", "Hybrid-S2", "o", (-58, -12)), "Hybrid-A2": ("#1baf7a", "Hybrid-A2", "o", (-60, 6)),
        "Hybrid-S3": ("#1baf7a", "Hybrid-S3", "D", (8, -4)), "Hybrid-A3": ("#1baf7a", "Hybrid-A3", "D", (8, 4))}
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for m, (c, lb, mk, off) in show.items():
    r = summ.loc[m]
    ax.plot(r.mean_auroc, r.mean_bal_acc, mk, color=c, ms=9 if mk == "o" else 8)
    ax.annotate(lb, (r.mean_auroc, r.mean_bal_acc), textcoords="offset points", xytext=off, fontsize=8.5, color="#0b0b0b")
for a, b in [("CBM2-Sign", "CBM3-Sign"), ("CBM2-SignAda", "CBM3-SignAda"), ("Hybrid-S2", "Hybrid-S3"), ("Hybrid-A2", "Hybrid-A3")]:
    ax.annotate("", xy=(summ.loc[b].mean_auroc, summ.loc[b].mean_bal_acc), xytext=(summ.loc[a].mean_auroc, summ.loc[a].mean_bal_acc),
                arrowprops=dict(arrowstyle="->", color="#9a9993", lw=1))
ax.set_xlabel("AUROC trung bình ngoài miền (9 thiết lập)"); ax.set_ylabel("Balanced accuracy trung bình (ngưỡng chọn trên train)")
ax.grid(color="#e4e3df"); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.set_title("Đánh đổi phân biệt – ngưỡng (mũi tên: điểm J 9 chuyển đạo → chỉ V1–V3)", loc="left", fontsize=10.5, color="#0b0b0b")
plt.tight_layout(); plt.savefig(os.path.join(OD, "fig_v4_tradeoff.png"), dpi=150); plt.close()

# kiến trúc
arch_p = os.path.join(OD, "arch_v4.json")
if os.path.exists(arch_p):
    A = pd.DataFrame(json.load(open(arch_p))).set_index(["train", "test"]).loc[ORDER]
    cnn9 = res[res.model == "CNN"].set_index(["train", "test"]).loc[ORDER]
    v1v3 = pd.DataFrame([r for r in json.load(open(os.path.join(OD, "v4_raw.json"))) if r["model"] == "CNN-V1V3" and r["vseed"] == 0]).set_index(["train", "test"]).loc[ORDER]
    v3raw = pd.DataFrame([r for r in json.load(open(os.path.join(V3, "v3_raw.json"))) if r["model"] == "CNN" and r["vseed"] == 0]).set_index(["train", "test"]).loc[ORDER]
    A["CNN-9"] = v3raw.auroc.round(3).values; A["CNN-V1V3"] = v1v3.auroc.round(3).values
    print(A.to_string()); print(A.mean().round(3).to_string())
    A.to_csv(os.path.join(OD, "arch_v4.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, (arch, c) in enumerate([("LogReg", "#9a9993"), ("CNN", "#eb6834"), ("ResNet", "#6250d6")]):
        a9, a3 = A[f"{arch}-9"].values, A[f"{arch}-V1V3"].values
        for i in range(len(a9)):
            ax.plot([k - 0.15, k + 0.15], [a9[i], a3[i]], color=c, alpha=0.35, lw=1)
        ax.plot([k - 0.15, k + 0.15], [a9.mean(), a3.mean()], "-o", color=c, lw=3, ms=8)
        ax.annotate(f"{a9.mean():.3f} → {a3.mean():.3f}", (k, max(a3.mean(), a9.mean()) + 0.012), ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(["LogReg", "CNN", "ResNet1D"])
    ax.set_ylabel("AUROC ngoài miền (trái: 9 chuyển đạo, phải: V1–V3)")
    ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Hiệu ứng 'chỉ V1–V3' trên 3 kiến trúc (đường mờ: từng thiết lập)", loc="left", fontsize=10.5, color="#0b0b0b")
    plt.tight_layout(); plt.savefig(os.path.join(OD, "fig_v4_arch.png"), dpi=150); plt.close()
