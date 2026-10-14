"""Phân tích bước 6.
  (1) CATT so với các phương pháp không nhãn trong y văn (CNN-V1V3), theo tỉ lệ bệnh, tách phát triển / held-out
  (2) CATT với các bộ gán nhãn giả khác nhau (khái niệm so với hộp đen) + cơ chế (α, β, phụ thuộc có điều kiện)
  (3) Hình: fig_v6_methods.png, fig_v6_labelers.png

Chạy: python analyze_v6.py <concepts_v6>
"""
import glob
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OD = sys.argv[1]
pd.set_option("display.width", 250)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#9a9993", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
                     "axes.spines.top": False, "axes.spines.right": False})
SHORT = {"huca": "H", "claris": "C", "mortara": "M"}
ROWS = [json.loads(l) for f in sorted(glob.glob(os.path.join(OD, "v6_raw_*.jsonl"))) for l in open(f)]
assert len(ROWS) == 27, len(ROWS)
print("tái lập CNN-V1V3 so với bước 5: chênh lớn nhất", max(r["repro_maxdiff"] for r in ROWS))
dfs = []
for r in ROWS:
    d = pd.DataFrame(r["res"], columns=r["cols"])
    d["setting"] = r["setting"]; d["vseed"] = r["vseed"]
    d["name"] = "+".join(SHORT[a] for a in r["train"].split("+")) + "→" + SHORT[r["test"]]
    dfs.append(d)
D = pd.concat(dfs, ignore_index=True)
D["prev"] = D["prev"].astype(str)
D["split"] = np.where(D.setting.isin([0, 1]), "phát triển", "held-out")
PREV = ["0.05", "0.1", "0.2", "tự nhiên"]
BASE = ["SRC", "EM-prior", "BBSE-Q", "MED", "AdaBN", "TENT"]
LAB = ["CBM5", "CBM3", "ST_V2", "CNN9", "LogReg9", "ResNet", "CNNalt"]
CONCEPT = {"CBM5", "CBM3", "ST_V2"}
MAIN = BASE + ["CATT[CBM5]", "AdaBN+CATT", "ORACLE"]
G = D.groupby(["split", "setting", "name", "method", "prev"])[["bacc", "sens", "spec"]].mean().reset_index()
S = G.groupby(["split", "method", "prev"]).agg(bacc=("bacc", "mean"), bacc_min=("bacc", "min"), sens=("sens", "mean"),
                                              spec=("spec", "mean")).reset_index()
S.round(3).to_csv(os.path.join(OD, "v6_summary.csv"), index=False)
for split in ["phát triển", "held-out"]:
    t = S[S.split == split].pivot(index="method", columns="prev", values="bacc")
    tm = S[S.split == split].pivot(index="method", columns="prev", values="bacc_min")
    order = MAIN + [f"CATT[{k}]" for k in LAB if k != "CBM5"]
    print(f"\n[{split}] CNN-V1V3: balanced accuracy TB (thấp nhất)")
    print((t.loc[order, PREV].round(3).astype(str) + " (" + tm.loc[order, PREV].round(3).astype(str) + ")").to_string())
# kiểm định trên held-out: CATT[CBM5] so với từng phương pháp
tests = []
for p in PREV:
    h = G[(G.split == "held-out") & (G.prev == p)].pivot(index="setting", columns="method", values="bacc")
    for m in [x for x in h.columns if x not in ("CATT[CBM5]", "ORACLE")]:
        d = h["CATT[CBM5]"] - h[m]
        pv = wilcoxon(h["CATT[CBM5]"], h[m]).pvalue if (d.abs() > 1e-9).any() else 1.0
        tests.append(dict(prev=p, vs=m, mean_diff=round(float(d.mean()), 3), catt_better=int((d > 0.01).sum()),
                          catt_worse=int((d < -0.01).sum()), wilcoxon_p=round(float(pv), 4)))
T = pd.DataFrame(tests)
T.to_csv(os.path.join(OD, "v6_tests_heldout.csv"), index=False)
print("\nHeld-out (7 thiết lập), CATT[CBM5] trừ phương pháp khác:")
print(T[T.prev.isin(["0.05", "tự nhiên"])].to_string(index=False))
# tất cả 9 thiết lập, theo từng thiết lập (tự nhiên)
per = G[(G.prev == "tự nhiên") & G.method.isin(["SRC", "EM-prior", "MED", "AdaBN", "TENT", "CATT[CBM5]", "CATT[CNN9]", "CATT[CNNalt]", "ORACLE"])]
per = per.pivot_table(index=["setting", "name"], columns="method", values="bacc").round(3)
print("\nTừng thiết lập (tỉ lệ tự nhiên):"); print(per.to_string())
per.to_csv(os.path.join(OD, "v6_per_setting_natural.csv"))
# ---------------- cơ chế
M = []
for r in ROWS:
    for k, v in r["mech"].items():
        M.append(dict(setting=r["setting"], vseed=r["vseed"], labeler=k, **v))
M = pd.DataFrame(M)
gain = G[G.method.str.startswith("CATT[")].copy()
gain["labeler"] = gain.method.str[5:-1]
oracle = G[G.method == "ORACLE"].set_index(["setting", "prev"]).bacc
gain["gap"] = oracle.loc[list(zip(gain.setting, gain.prev))].values - gain.bacc
Mm = M.groupby(["setting", "labeler"])[["alpha", "beta", "one_minus", "cond_rho", "auroc"]].mean().reset_index()
J = Mm.merge(gain[gain.prev == "tự nhiên"][["setting", "labeler", "bacc", "gap"]], on=["setting", "labeler"])
lab_tab = J.groupby("labeler")[["auroc", "alpha", "beta", "one_minus", "cond_rho", "bacc", "gap"]].mean().loc[LAB]
lab_tab["bacc_5%"] = gain[gain.prev == "0.05"].groupby("labeler").bacc.mean().loc[LAB]
lab_tab["bacc_min"] = J.groupby("labeler").bacc.min().loc[LAB]
lab_tab["held_out_bacc"] = gain[(gain.prev == "tự nhiên") & (gain.split == "held-out")].groupby("labeler").bacc.mean().loc[LAB]
lab_tab["loại"] = ["khái niệm" if k in CONCEPT else "hộp đen" for k in LAB]
print("\nBộ gán nhãn giả (TB 9 thiết lập, tỉ lệ tự nhiên trừ cột 5%):"); print(lab_tab.round(3).to_string())
lab_tab.round(3).to_csv(os.path.join(OD, "v6_labelers.csv"))
r1 = spearmanr(J.cond_rho, J.gap); r2 = spearmanr(J.one_minus, J.gap); r3 = spearmanr(J.auroc, J.gap)
mech_stats = dict(spearman_gap_vs_cond_rho=[round(r1[0], 3), float(r1[1])], spearman_gap_vs_one_minus=[round(r2[0], 3), float(r2[1])],
                  spearman_gap_vs_labeler_auroc=[round(r3[0], 3), float(r3[1])], n_points=len(J))
from sklearn.linear_model import LinearRegression  # noqa: E402


def zs(v):
    return (v - v.mean()) / v.std()


mech_stats["spearman_cond_rho_vs_one_minus"] = round(float(spearmanr(J.cond_rho, J.one_minus)[0]), 3)
for name, cols in [("linreg_theory", ["cond_rho", "one_minus"]), ("linreg_plus_auroc", ["cond_rho", "one_minus", "auroc"])]:
    Xr = np.c_[[zs(J[c]) for c in cols]].T
    lr = LinearRegression().fit(Xr, J.gap)
    mech_stats[name] = dict(std_coef={c: round(float(v), 4) for c, v in zip(cols, lr.coef_)}, r2=round(float(lr.score(Xr, J.gap)), 3))
print("\nCơ chế (63 điểm = 9 thiết lập × 7 bộ gán nhãn):", mech_stats)
json.dump(mech_stats, open(os.path.join(OD, "v6_mechanism.json"), "w"), indent=1, ensure_ascii=False)
J.round(4).to_csv(os.path.join(OD, "v6_mechanism_points.csv"), index=False)
ad = pd.DataFrame([dict(setting=r["setting"], **r["auroc_adapt"]) for r in ROWS]).groupby("setting").mean()
print("\nAUROC của CNN-V1V3 trước/sau thích nghi lúc test (TB 9 thiết lập):", ad.mean().round(3).to_dict())
ad.round(3).to_csv(os.path.join(OD, "v6_auroc_adapt.csv"))
pi = D[D.method.isin(["EM-prior", "BBSE-Q"]) & (D.prev != "tự nhiên")].copy()
pi["true"] = pi.prev.astype(float)
print("\nƯớc lượng tỉ lệ bệnh (TB):"); print(pi.groupby(["method", "prev"]).pi_hat.mean().round(3).unstack())

# ---------------- hình
COL = {"SRC": "#9a9993", "EM-prior": "#c9a227", "BBSE-Q": "#b07cc6", "MED": "#e0a33a", "AdaBN": "#5aa5d6", "TENT": "#2f6fa8",
       "CATT[CBM5]": "#eb6834", "AdaBN+CATT": "#f2b39a", "ORACLE": "#1baf7a"}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
xs = np.arange(4)
for k, split in enumerate(["held-out", "phát triển"]):
    for m in ["SRC", "EM-prior", "BBSE-Q", "MED", "AdaBN", "TENT", "CATT[CBM5]", "ORACLE"]:
        t = S[(S.split == split) & (S.method == m)].set_index("prev").loc[PREV]
        ax[k].plot(xs, t.bacc, marker="o", color=COL[m], lw=2.6 if m == "CATT[CBM5]" else 1.5, ls=":" if m == "ORACLE" else "-",
                   label={"CATT[CBM5]": "CATT (đề xuất)", "ORACLE": "ORACLE (dùng nhãn đích)"}.get(m, m))
    ax[k].set_xticks(xs); ax[k].set_xticklabels(["5%", "10%", "20%", "tự nhiên"]); ax[k].set_xlabel("tỉ lệ bệnh ở miền đích")
    ax[k].set_title(f"CNN-V1V3, {'7 thiết lập held-out' if split == 'held-out' else '2 thiết lập phát triển'}", fontsize=10, loc="left")
    ax[k].grid(alpha=0.3)
ax[0].set_ylabel("balanced accuracy TB"); ax[0].legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(OD, "fig_v6_methods.png"), dpi=150); plt.close()

LC = {"CBM5": "#eb6834", "CBM3": "#f28c5c", "ST_V2": "#f5b08f", "CNN9": "#5aa5d6", "LogReg9": "#8fc1e3", "ResNet": "#6250d6", "CNNalt": "#2b2a7a"}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
lt = lab_tab.copy()
yy = np.arange(len(LAB))
ax[0].barh(yy, lt.bacc, color=[LC[k] for k in LAB])
ax[0].scatter(lt.bacc_min, yy, color="#52514e", marker="|", s=200, zorder=3, label="thấp nhất")
ax[0].axvline(G[(G.method == "SRC") & (G.prev == "tự nhiên")].bacc.mean(), color="#9a9993", ls="--", lw=1.2, label="SRC")
ax[0].axvline(G[(G.method == "ORACLE") & (G.prev == "tự nhiên")].bacc.mean(), color="#1baf7a", ls=":", lw=1.5, label="ORACLE")
ax[0].set_yticks(yy); ax[0].set_yticklabels([f"{k} ({'khái niệm' if k in CONCEPT else 'hộp đen'})" for k in LAB]); ax[0].invert_yaxis()
ax[0].set_xlim(0.5, 0.85); ax[0].set_xlabel("balanced accuracy TB của CNN-V1V3 sau CATT (9 thiết lập)")
ax[0].set_title("a. CATT với từng bộ gán nhãn giả", fontsize=10, loc="left"); ax[0].legend(fontsize=8, frameon=False, loc="lower right")
for k in LAB:
    j = J[J.labeler == k]
    ax[1].scatter(j.cond_rho, j.gap, color=LC[k], s=28, label=k, alpha=0.85)
    ax[2].scatter(j.one_minus, j.gap, color=LC[k], s=28, alpha=0.85)
ax[1].set_xlabel("phụ thuộc có điều kiện với CNN-V1V3 (Spearman trong lớp)"); ax[1].set_ylabel("khoảng cách tới ORACLE")
ax[1].set_title(f"b. Lỗi càng phụ thuộc, CATT càng kém (ρ = {r1[0]:.2f})", fontsize=10, loc="left"); ax[1].legend(fontsize=7.5, frameon=False, ncol=2)
ax[2].set_xlabel("1 − α − β của nhãn giả (định lý Menon 2015)")
ax[2].set_title(f"c. Nhãn giả càng cân, CATT càng tốt (ρ = {r2[0]:.2f})", fontsize=10, loc="left")
for a in ax[1:]:
    a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OD, "fig_v6_labelers.png"), dpi=150); plt.close()
print("xong")
