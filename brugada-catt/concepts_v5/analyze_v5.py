"""Phân tích bước 5.
  (1) AUROC ngoài miền: CBM3 (QRS_w) / CBM3n (bỏ QRS) / CBM5 (QRS_sd15) / Hybrid; so sánh cặp bootstrap + Holm
  (2) Hiệu chỉnh ngưỡng không nhãn (recal_raw.csv): balanced accuracy theo phương pháp × tỉ lệ bệnh,
      tách tập phát triển (setting 0, 1) và held-out (setting 2–8); Wilcoxon trên held-out
  (3) Hình: fig_v5_qrs.png (khái niệm QRS), fig_v5_recal.png (hiệu chỉnh ngưỡng)

Chạy: python analyze_v5.py <concepts_v5> <harmonized>
"""
import glob
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OD, HD = sys.argv[1], sys.argv[2]
Y = {n: np.load(os.path.join(HD, f"{n}.npz"))["y"].astype(int) for n in ["huca", "claris", "mortara"]}
ORDER = [("huca", "claris"), ("huca", "mortara"), ("claris", "huca"), ("claris", "mortara"), ("mortara", "huca"),
         ("mortara", "claris"), ("claris+mortara", "huca"), ("huca+mortara", "claris"), ("huca+claris", "mortara")]
SHORT = {"huca": "H", "claris": "C", "mortara": "M"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#9a9993", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
                     "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.width", 250)


def lab(tr, te):
    return "+".join(SHORT[a] for a in tr.split("+")) + "→" + SHORT[te]


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


# ------------------------------------------------------------------ (1) AUROC
ROWS = [json.loads(l) for f in sorted(glob.glob(os.path.join(OD, "v5_raw_*.jsonl"))) for l in open(f)]
MODELS = list(ROWS[0]["models"].keys())
S, res = {}, []
for tr, te in ORDER:
    rs = [r for r in ROWS if r["train"] == tr and r["test"] == te]
    for m in MODELS:
        s = np.mean([np.array(r["models"][m]["tgt"]) for r in rs], 0); S[(tr, te, m)] = s
        res.append(dict(train=tr, test=te, setting=lab(tr, te), model=m, n_vseed=len(rs), auroc=round(fast_auc(Y[te], s), 3),
                        seed_sd=round(float(np.std([r["models"][m]["auroc"] for r in rs])), 3)))
res = pd.DataFrame(res)
tab = res.pivot_table(index="setting", columns="model", values="auroc", sort=False)[MODELS]
tab.loc["TRUNG BÌNH"] = tab.mean()
tab.loc["THẤP NHẤT"] = tab.iloc[:-1].min()
print(tab.round(3).to_string())
tab.round(3).to_csv(os.path.join(OD, "v5_auroc.csv"))
comps = []
for m, ref in [("CBM5-Sign", "CBM3-Sign"), ("CBM3n-Sign", "CBM3-Sign"), ("CBM5-Sign", "CBM3n-Sign"), ("CBM5-SignAda", "CBM5-Sign"),
               ("Hybrid-S5", "Hybrid-S3"), ("Hybrid-A5", "Hybrid-S3"), ("Hybrid-S3", "CNN-V1V3"), ("Hybrid-A5", "CNN-V1V3")]:
    block = []
    for tr, te in ORDER:
        d, lo, hi, p = boot_delta(Y[te], S[(tr, te, m)], S[(tr, te, ref)])
        block.append(dict(model=m, ref=ref, setting=lab(tr, te), delta=round(d, 3), lo=round(lo, 3), hi=round(hi, 3), p=p))
    for b, a in zip(block, holm([b["p"] for b in block])):
        b["p_holm"] = round(float(a), 4)
        b["verdict"] = "thắng" if (a < 0.05 and b["delta"] > 0) else ("thua" if (a < 0.05 and b["delta"] < 0) else "hòa")
    comps += block
comp = pd.DataFrame(comps)
comp.to_csv(os.path.join(OD, "v5_paired_holm.csv"), index=False)
wl = comp.groupby(["model", "ref"], sort=False).agg(mean_delta=("delta", "mean"), thang=("verdict", lambda v: (v == "thắng").sum()),
                                                    hoa=("verdict", lambda v: (v == "hòa").sum()), thua=("verdict", lambda v: (v == "thua").sum()))
print(wl.round(3).to_string())
wl.round(3).to_csv(os.path.join(OD, "v5_paired_summary.csv"))
# cổng dấu: QRS có qua cổng dấu không
sg = []
for r in ROWS:
    for m in ["CBM3-Sign", "CBM5-Sign"]:
        sg.append(dict(model=m, setting=lab(r["train"], r["test"]), qrs_sign=r["signs"][m][-1]))
sg = pd.DataFrame(sg).groupby(["model", "setting"], sort=False).qrs_sign.mean().unstack(0)
print("dấu của khái niệm QRS sau cổng dấu (trung bình 3 seed; 1 = luôn qua, 0 = bị loại):"); print(sg.to_string())
sg.to_csv(os.path.join(OD, "v5_qrs_sign_gate.csv"))

# ------------------------------------------------------------------ (2) hiệu chỉnh ngưỡng
rc = pd.read_csv(os.path.join(OD, "recal_raw.csv"))
rc["split"] = np.where(rc.setting.isin([0, 1]), "phát triển", "held-out")
rc["prev"] = rc["prev"].astype(str)
PREV_ORDER = ["0.05", "0.1", "0.2", "tự nhiên"]
METH = ["SRC", "MED", "LSM", "CATT", "CATT-LSM", "ORACLE"]
g = rc.groupby(["split", "setting", "model", "method", "prev"])[["bacc", "sens", "spec", "pi_hat", "true_prev"]].mean().reset_index()
summ = g.groupby(["split", "model", "method", "prev"]).agg(bacc=("bacc", "mean"), bacc_min=("bacc", "min"), sens=("sens", "mean"),
                                                          spec=("spec", "mean")).reset_index()
summ.round(3).to_csv(os.path.join(OD, "recal_summary.csv"), index=False)
for split in ["phát triển", "held-out"]:
    for m in ["CNN-V1V3", "Hybrid-S3", "Hybrid-S5", "Hybrid-A5", "CBM5-Sign"]:
        t = summ[(summ.split == split) & (summ.model == m)].pivot(index="method", columns="prev", values="bacc").loc[METH, PREV_ORDER]
        tm = summ[(summ.split == split) & (summ.model == m)].pivot(index="method", columns="prev", values="bacc_min").loc[METH, PREV_ORDER]
        print(f"\n[{split}] {m}: balanced accuracy TB (thấp nhất)")
        print((t.round(3).astype(str) + " (" + tm.round(3).astype(str) + ")").to_string())
tests = []
for m in ["CNN-V1V3", "Hybrid-S3", "Hybrid-S5", "Hybrid-A5", "CBM5-Sign", "CBM3-Sign"]:
    for p in PREV_ORDER:
        h = g[(g.split == "held-out") & (g.model == m) & (g.prev == p)].pivot(index="setting", columns="method", values="bacc")
        for meth in ["MED", "LSM", "CATT", "CATT-LSM"]:
            d = h[meth] - h["SRC"]
            pv = wilcoxon(h[meth], h["SRC"]).pvalue if (d != 0).any() else 1.0
            tests.append(dict(model=m, prev=p, method=meth, mean_gain=round(float(d.mean()), 3), n_better=int((d > 0.01).sum()),
                              n_worse=int((d < -0.01).sum()), wilcoxon_p=round(float(pv), 4),
                              gap_to_oracle=round(float((h["ORACLE"] - h[meth]).mean()), 3)))
tests = pd.DataFrame(tests)
tests.to_csv(os.path.join(OD, "recal_heldout_tests.csv"), index=False)
print("\nHeld-out (7 thiết lập), so với SRC:"); print(tests[tests.method.isin(["CATT", "MED"])].to_string(index=False))
scr = g[(g.prev == "0.05") & (g.method.isin(["SRC", "CATT", "ORACLE"]))].groupby(["model", "method"])[["sens", "spec", "bacc"]].mean()
print("\nSàng lọc, tỉ lệ bệnh 5% (cả 9 thiết lập):"); print(scr.round(3).to_string())
scr.round(3).to_csv(os.path.join(OD, "recal_screening5.csv"))
per_set = g[(g.method.isin(["SRC", "CATT", "ORACLE"])) & (g.prev == "tự nhiên")].pivot_table(
    index=["setting", "model"], columns="method", values=["sens", "spec"]).round(3)
per_set.to_csv(os.path.join(OD, "recal_per_setting_natural.csv"))

# ------------------------------------------------------------------ (3) hình
Q = json.load(open(os.path.join(OD, "qrs_test5.json")))
COL = {"abs": "#9a9993", "rel": "#e0a33a", "lp15": "#5aa5d6", "sd15": "#6250d6"}
NAME = {"abs": "QRS_w (cũ)", "rel": "QRS_rel (chọn trước, loại)", "lp15": "QRS_w15 (loại)", "sd15": "QRS_sd15 (chọn)"}
fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
ks = np.array(Q["Q3_k"])
for w in ["abs", "rel", "sd15"]:
    v = np.array(Q["Q3_widening"][w]["local"]); u = np.array(Q["Q3_widening"][w]["glob"])
    ax[0].plot(ks, (v - v[0]) / (np.ptp(v) + 1e-9), color=COL[w], lw=2, label=f"{NAME[w]}: giãn khu trú V1–V3")
    ax[0].plot(ks, (u - u[0]) / (np.ptp(v) + 1e-9), color=COL[w], lw=1.5, ls="--")
ax[0].set_xlabel("hệ số giãn QRS (ECG giả lập)"); ax[0].set_ylabel("thay đổi khái niệm (chuẩn hóa)")
ax[0].set_title("a. Phantom: liền = giãn V1–V3, đứt = giãn mọi chuyển đạo", fontsize=10, loc="left")
ax[0].legend(fontsize=7.5, frameon=False)
pert = ["lowpass30", "lowpass25", "lowpass20", "smooth3"]
xx = np.arange(len(pert)); W = 0.2
for i, w in enumerate(["abs", "rel", "lp15", "sd15"]):
    vals = [np.mean([abs(Q["Q4_device_perturbation_SMD"][p][d][w]) for d in ["huca", "claris", "mortara"]]) for p in pert]
    ax[1].bar(xx + (i - 1.5) * W, vals, W, color=COL[w], label=NAME[w])
ax[1].set_xticks(xx); ax[1].set_xticklabels(["lọc 30 Hz", "25 Hz", "20 Hz", "trơn 3 điểm"])
ax[1].set_ylabel("|SMD| do nhiễu loạn (nhóm âm tính)"); ax[1].legend(fontsize=7.5, frameon=False)
ax[1].set_title("b. Nhiễu loạn kiểu thiết bị mô phỏng", fontsize=10, loc="left")
pairs = list(Q["Q5_real"]["shift_neg_SMD"].keys())
for i, w in enumerate(["abs", "rel", "lp15", "sd15"]):
    sh = [abs(Q["Q5_real"]["shift_neg_SMD"][p][w]) for p in pairs]
    au = np.mean([Q["Q5_real"]["auroc"][d][w] for d in ["huca", "claris", "mortara"]])
    ax[2].scatter(np.mean(sh), au, s=90, color=COL[w], label=NAME[w], zorder=3)
    ax[2].annotate(f"max |SMD| {max(sh):.2f}", (np.mean(sh), au), textcoords="offset points", xytext=(6, -12), fontsize=7.5, color="#52514e")
ax[2].set_xlabel("|SMD| TB giữa các miền thật (nhóm âm tính)"); ax[2].set_ylabel("AUROC TB trong từng miền (1 khái niệm)")
ax[2].set_title("c. Dữ liệu thật, khái niệm đứng riêng", fontsize=10, loc="left"); ax[2].grid(alpha=0.3)
INFO = json.load(open(os.path.join(OD, "qrs_info5.json")))
cbm = {"abs": tab.loc["TRUNG BÌNH", "CBM3-Sign"], "none": tab.loc["TRUNG BÌNH", "CBM3n-Sign"], "sd15": tab.loc["TRUNG BÌNH", "CBM5-Sign"]}
labels = [("abs", "QRS_w"), ("sd15", "QRS_sd15"), ("none", "bỏ QRS")]
for i, (w, nm) in enumerate(labels):
    ax[3].bar(i, cbm[w], 0.6, color=COL.get(w, "#d6d5d0"))
    txt = f"{cbm[w]:.3f}"
    if w != "none":
        txt += f"\nAUROC phần dư {np.mean([INFO['I1'][d][w]['auroc_residual'] for d in ['huca', 'claris', 'mortara']]):.2f}"
    ax[3].text(i, cbm[w] + 0.002, txt, ha="center", va="bottom", fontsize=8, color="#52514e")
ax[3].set_xticks(range(3)); ax[3].set_xticklabels([n for _, n in labels]); ax[3].set_ylim(0.78, 0.86)
ax[3].set_ylabel("AUROC ngoài miền TB của CBM (9 thiết lập)")
ax[3].set_title("d. Trong CBM: QRS_w mang thông tin riêng", fontsize=10, loc="left")
plt.tight_layout(); plt.savefig(os.path.join(OD, "fig_v5_qrs.png"), dpi=150); plt.close()

MC = {"SRC": "#9a9993", "MED": "#e0a33a", "LSM": "#5aa5d6", "CATT": "#eb6834", "CATT-LSM": "#f2b39a", "ORACLE": "#1baf7a"}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
xs = np.arange(4)
for k, m in enumerate(["CNN-V1V3", "Hybrid-S3", "Hybrid-A5"]):
    for meth in ["SRC", "MED", "LSM", "CATT", "ORACLE"]:
        t = summ[(summ.split == "held-out") & (summ.model == m) & (summ.method == meth)].set_index("prev").loc[PREV_ORDER]
        ax[k].plot(xs, t.bacc, marker="o", color=MC[meth], lw=2.4 if meth == "CATT" else 1.6, ls=":" if meth == "ORACLE" else "-",
                   label=meth + (" (đề xuất)" if meth == "CATT" else (" (dùng nhãn đích)" if meth == "ORACLE" else "")))
        ax[k].fill_between(xs, t.bacc_min, t.bacc, color=MC[meth], alpha=0.06) if meth in ("SRC", "CATT") else None
    ax[k].set_xticks(xs); ax[k].set_xticklabels(["5%", "10%", "20%", "tự nhiên"]); ax[k].set_xlabel("tỉ lệ bệnh ở miền đích")
    ax[k].set_title(f"{m} (7 thiết lập held-out)", fontsize=10, loc="left"); ax[k].grid(alpha=0.3)
ax[0].set_ylabel("balanced accuracy TB (dải mờ: tới mức thấp nhất)"); ax[0].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OD, "fig_v5_recal.png"), dpi=150); plt.close()
print("xong")
