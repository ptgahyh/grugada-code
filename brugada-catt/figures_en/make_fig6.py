"""Figure 6 (step 8): (a) simulation of CATT bias, (b) unlabelled target ECGs needed. Uses styles of make_figures.py.
Run: python make_fig6.py ../analysis_v8"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_figures.py")).read().split("# ----")[0]
sys.argv = [sys.argv[0], ".."] + sys.argv[1:]
A8 = sys.argv[2] if len(sys.argv) > 2 else "../analysis_v8"
exec(src)
S = pd.read_csv(os.path.join(A8, "sim_catt_degenerate.csv"))
C = pd.read_csv(os.path.join(A8, "v8_sample_size_curve.csv"))
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
ax = axes[0]
for rho, col, lab in [(0.0, BLUE, "independent (ρ = 0)"), (0.9, ORANGE, "dependent (ρ = 0.9)")]:
    for n, ls in [(356, "-"), (1154, (0, (3, 2)))]:
        d = S[(S.rho == rho) & (S.n == n)].sort_values("one_minus_alpha_beta")
        ax.plot(d.one_minus_alpha_beta, d.gap_mean, color=col, ls=ls, lw=1.8, marker="o", markersize=3.5,
                label=f"{lab}, n = {n}")
ax.set_xlabel("Separation factor 1 − α − β of the pseudo-labels"); ax.set_ylabel("Gap to oracle (balanced accuracy)")
ax.set_title("Dependence only hurts when the labeller's\nthreshold is off, and does not shrink with n", loc="left")
ax.legend(frameon=False, fontsize=7, loc="upper right"); hgrid(ax, "y"); ax.set_ylim(0, 0.16)
panel(ax, "a")
ax = axes[1]
H = C[C.setting.isin([2, 3, 4, 5, 6, 7, 8])].copy()
H["nn"] = H.n.replace("all", "600").astype(int)
G = H.groupby(["nn", "method"])[["mean", "p10"]].mean().reset_index()
order = [20, 30, 50, 75, 100, 150, 200, 300, 600]
x = np.arange(len(order))
for m, col, lw in [("SRC", GREY_L, 1.4), ("MED", GREY, 1.3), ("UPA", GREY, 1.3), ("GMM-2", GREY, 1.3), ("CATT", BLUE, 2.2), ("ORACLE", INK, 1.2)]:
    g = G[G.method == m].set_index("nn").loc[order]
    ls = {"UPA": (0, (3, 1.5)), "GMM-2": (0, (1, 1.2)), "ORACLE": (0, (4, 2))}.get(m, "-")
    ax.plot(x, g["mean"], color=col, lw=lw, ls=ls, marker="o", markersize=3 if m != "CATT" else 4)
    if m == "CATT":
        ax.fill_between(x, g["p10"], g["mean"], color=BLUE, alpha=0.12, lw=0)
    ax.text(len(order) - 0.85, g["mean"].iloc[-1] + {"UPA": 0.003, "GMM-2": -0.003, "MED": -0.002}.get(m, 0),
            {"ORACLE": "Oracle"}.get(m, m), fontsize=7.2, va="center", color=BLUE if m == "CATT" else INK2,
            fontweight="semibold" if m == "CATT" else "normal")
ax.set_xticks(x); ax.set_xticklabels([str(v) for v in order[:-1]] + ["all"]); ax.set_xlim(-0.3, len(order) + 0.2)
ax.set_xlabel("Unlabelled target ECGs used to set the threshold"); ax.set_ylabel("Held-out balanced accuracy (7 settings)")
ax.set_title("About 50–100 unlabelled ECGs suffice;\nshading: 10th percentile of CATT", loc="left")
hgrid(ax, "y"); ax.set_ylim(0.68, 0.83)
panel(ax, "b")
fig.tight_layout(w_pad=2.0)
save(fig, "fig6_simulation_samplesize")
print("ok")
