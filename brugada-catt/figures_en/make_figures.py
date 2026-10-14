"""English, publication-style figures for the Brugada CATT manuscript.

Run: python make_figures.py <outputs dir containing concepts_v4, concepts_v5, concepts_v6>
Writes fig1..fig5 as PNG (300 dpi) and PDF next to this script.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = sys.argv[1] if len(sys.argv) > 1 else ".."
OUT = os.path.dirname(os.path.abspath(__file__))

for f in font_manager.findSystemFonts():
    if "Inter" in os.path.basename(f):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Inter", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.7, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.grid": False, "savefig.facecolor": "white", "figure.facecolor": "white", "pdf.fonttype": 42,
})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GREY, GREY_L, GRID = "#a3a29c", "#d6d5cf", "#ecebe6"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

SETTINGS = ["HUCA→Claris", "HUCA→Mortara", "Claris→HUCA", "Claris→Mortara", "Mortara→HUCA",
            "Mortara→Claris", "Claris+Mortara→HUCA", "HUCA+Mortara→Claris", "HUCA+Claris→Mortara"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def hgrid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")


# --------------------------------------------------------------------------- Figure 1: overview schematic
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")

    def box(x, y, w, h, title, body="", fc="#f6f6f4", ec=GREY, tc=INK, bc=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=8.2, fontweight="semibold", color=tc)
        if body:
            ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=7.2, color=bc, linespacing=1.3)

    def arrow(x0, y0, x1, y1, color=MUTED, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=color,
                                     lw=1.0, linestyle=ls, shrinkA=0, shrinkB=0))

    ax.text(0.5, 63.5, "A   Training on labelled source sites", fontsize=9, fontweight="bold", va="top")
    ax.text(0.5, 29.5, "B   Deployment at an unlabelled target site", fontsize=9, fontweight="bold", va="top")
    ax.plot([0.5, 99.5], [31.2, 31.2], color=GRID, lw=1.0)

    box(1, 42, 17, 14, "3 BrS cohorts", "HUCA, Claris,\nMortara\n1,880 ECGs")
    box(23, 42, 17, 14, "Harmonize", "9 leads · 100 Hz\nmedian beat\nz-scored")
    arrow(18.6, 49, 22.4, 49)
    box(46, 50, 22, 10, "Target model f", "CNN on V1–V3")
    box(46, 33, 22, 15, "Concept model g", "soft J point →\n28 concepts → sign-\nconstrained CBM",
        fc="#eef4fc", ec=BLUE, tc=BLUE)
    arrow(40.6, 52, 45.4, 55); arrow(40.6, 46, 45.4, 41)
    ax.text(31.5, 40, "Youden thresholds\n$t_f$, $t_g$ on source\nvalidation", fontsize=6.9, color=INK2, ha="center", va="top")
    box(75, 33, 24, 15, "Physician-free checks", "synthetic phantoms\nJ point vs LUDB\n(no clinician labels)")
    arrow(74.4, 40.5, 68.6, 40.5, color=BLUE, ls=(0, (2, 1.5)))

    box(1, 8, 17, 14, "Target batch", "unlabelled ECGs\nfrom a new site")
    box(24, 15.5, 20, 9.5, "Scores f(x)", "shifted: $t_f$ misplaced")
    box(24, 2, 20, 10.5, "Pseudo-labels", "ỹ = 1[g(x) ≥ $t_g$]", fc="#eef4fc", ec=BLUE, tc=BLUE)
    arrow(18.6, 17, 23.4, 20); arrow(18.6, 12, 23.4, 8, color=BLUE)
    box(50, 2.5, 21, 21.5, "CATT threshold", "Youden of f(x)\nagainst ỹ\n\nno target labels,\nf unchanged",
        fc=BLUE, ec=BLUE, tc="white", bc="#e3eefb")
    arrow(44.6, 20, 49.4, 17); arrow(44.6, 7, 49.4, 10, color=BLUE)
    ax.text(75, 22, "Valid when the errors of g are\nconditionally independent of f\n(a second view) and α + β < 1:",
            fontsize=7.2, color=INK2, ha="left", va="top", linespacing=1.3)
    ax.text(75, 10.5, "$\\tilde J(t) \\propto (1-\\alpha-\\beta)\\,J(t)$", fontsize=8.5, color=INK, ha="left", va="top")
    save(fig, "fig1_overview")


# --------------------------------------------------------------------------- Figure 2: lead restriction
def fig2():
    a = pd.read_csv(os.path.join(ROOT, "concepts_v4", "arch_v4.csv"))
    pairs = [("Logistic regression", "LogReg-9", "LogReg-V1V3"), ("CNN", "CNN-9", "CNN-V1V3"), ("1D ResNet", "ResNet-9", "ResNet-V1V3")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True)
    y = np.arange(len(SETTINGS))[::-1]
    for k, (ax, (name, c9, c3)) in enumerate(zip(axes, pairs)):
        v9, v3 = a[c9].values, a[c3].values
        for yi, p, q in zip(y, v9, v3):
            ax.plot([p, q], [yi, yi], color=BLUE if q > p else GREY, lw=1.6, zorder=2, solid_capstyle="round")
        ax.scatter(v9, y, s=26, color="white", edgecolor=GREY, linewidth=1.3, zorder=3, label="9 leads")
        ax.scatter(v3, y, s=26, color=BLUE, edgecolor="white", linewidth=0.8, zorder=4, label="V1–V3 only")
        ax.set_title(name, loc="left", fontsize=9, fontweight="semibold", color=INK, pad=16)
        ax.text(0, 1.015, f"mean {v9.mean():.3f} → {v3.mean():.3f}, better in {(v3 > v9).sum()}/9",
                transform=ax.transAxes, fontsize=7.6, color=INK2, va="bottom")
        ax.set_xlim(0.66, 0.94); ax.set_xticks([0.7, 0.8, 0.9]); hgrid(ax)
        ax.set_xlabel("Out-of-domain AUROC")
        ax.spines["left"].set_visible(k == 0)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y); axes[0].set_yticklabels(SETTINGS)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.99, 1.03), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.2)
    fig.suptitle("Dropping six leads raises cross-site AUROC for every architecture", x=0.01, ha="left", y=1.02,
                 fontsize=10, fontweight="semibold")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig2_lead_restriction")


# --------------------------------------------------------------------------- Figure 3: LUDB J point
def fig3():
    d = pd.read_csv(os.path.join(ROOT, "concepts_v4", "ludb_J_per_record.csv"))
    d = d[~d.paced.astype(bool)]
    cond = d.conduction.astype(bool)
    rng = np.random.default_rng(0)
    groups = [("9-lead energy\ninitial", "err_v2-init"), ("V1–V3 energy\ninitial", "err_v3-init"),
              ("V1–V3 energy\nlearned by CBM", "err_v3-learned")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.axhspan(-10, 10, color="#f1f1ee", zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    for i, (lab, col) in enumerate(groups):
        for j, (mask, c, nm) in enumerate([(~cond, BLUE, "Normal conduction"), (cond, ORANGE, "Conduction disorder")]):
            v = d.loc[mask, col].values
            x = i + (j - 0.5) * 0.36 + rng.uniform(-0.1, 0.1, len(v))
            ax.scatter(x, np.clip(v, -60, 60), s=7, color=c, alpha=0.55, linewidth=0, zorder=2,
                       label=f"{nm} (n = {mask.sum()})" if i == 0 else None)
            med = np.median(v)
            ax.plot([i + (j - 0.5) * 0.36 - 0.14, i + (j - 0.5) * 0.36 + 0.14], [med, med], color=INK, lw=1.6, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(-62, 62); ax.set_ylabel("Soft J point − annotated QRS offset (ms)")
    ax.text(2.62, 10.5, "±10 ms", fontsize=7, color=MUTED, ha="right", va="bottom")
    ax.legend(loc="upper left", frameon=False, markerscale=2, handletextpad=0.2, borderaxespad=0.1)
    ax.set_title("At initialization the soft J point is unbiased (n = 191)", loc="left")
    ax.text(0.99, 0.01, "points beyond ±60 ms drawn at the edge", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=MUTED)
    panel(ax, "a")

    ax = axes[1]
    ok = d[~cond]
    ax.scatter(ok["ref_off_minus_R"], ok["pred_v3-init"], s=9, color=GREY, alpha=0.7, linewidth=0, label="initial")
    ax.scatter(ok["ref_off_minus_R"], ok["pred_v3-learned"], s=9, color=BLUE, alpha=0.8, linewidth=0, label="learned")
    lim = [20, 130]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(124, 117, "identity", fontsize=7, color=MUTED, ha="right")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Annotated QRS offset after R peak (ms)"); ax.set_ylabel("Soft J point after R peak (ms)")
    b0 = ok["err_v3-init"].mean(); b1 = ok["err_v3-learned"].mean()
    ax.set_title(f"Training moves it about 20 ms later\n(normal conduction; bias {b0:+.1f} → {b1:+.1f} ms)", loc="left")
    ax.legend(loc="upper left", frameon=False, markerscale=2, handletextpad=0.2)
    panel(ax, "b")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig3_ludb_jpoint")


# --------------------------------------------------------------------------- Figure 4: CATT
def fig4():
    p = pd.read_csv(os.path.join(ROOT, "concepts_v6", "v6_per_setting_natural.csv"))
    s = pd.read_csv(os.path.join(ROOT, "concepts_v6", "v6_summary.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(9)[::-1]
    for yi, (_, r) in zip(y, p.iterrows()):
        broken = r.ORACLE - r.SRC > 0.1
        ax.plot([r.SRC, r["CATT[CBM5]"]], [yi, yi], color=BLUE if broken else GREY_L, lw=1.8, zorder=2, solid_capstyle="round")
        ax.plot([r.ORACLE, r.ORACLE], [yi - 0.28, yi + 0.28], color=INK, lw=1.3, zorder=3)
    ax.scatter(p.SRC, y, s=26, color="white", edgecolor=GREY, linewidth=1.3, zorder=4, label="Source threshold (SRC)")
    ax.scatter(p["CATT[CBM5]"], y, s=28, color=BLUE, edgecolor="white", linewidth=0.8, zorder=5, label="CATT")
    ax.plot([], [], color=INK, lw=1.3, marker="|", ls="none", markersize=8, label="Oracle (target labels)")
    labels = [f"{n}  (dev)" if i < 2 else n for i, n in enumerate(SETTINGS)]
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0.5, 0.87); hgrid(ax); ax.set_xlabel("Balanced accuracy, CNN-V1V3 (natural prevalence)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8, fontsize=7.3)
    ax.set_title("CATT repairs the five broken settings\nand leaves the others alone", loc="left", pad=22)
    panel(ax, "a")

    ax = axes[1]
    prevs, xl = ["0.05", "0.1", "0.2", "tự nhiên"], ["5%", "10%", "20%", "natural"]
    h = s[s.split == "held-out"]
    meth = [("SRC", GREY_L, "-", 1.4), ("MED", GREY, "-", 1.4), ("AdaBN", GREY, (0, (3, 1.5)), 1.4),
            ("TENT", GREY, (0, (1, 1.2)), 1.4), ("CATT[CBM5]", BLUE, "-", 2.2), ("ORACLE", INK, (0, (4, 2)), 1.2)]
    x = np.arange(4)
    for m, c, ls, lw in meth:
        v = [h[(h.method == m) & (h.prev == q)].bacc.iloc[0] for q in prevs]
        ax.plot(x, v, color=c, ls=ls, lw=lw, marker="o", markersize=3.2 if m != "CATT[CBM5]" else 4, zorder=3 if m.startswith("CATT") else 2)
        name = {"CATT[CBM5]": "CATT", "ORACLE": "Oracle"}.get(m, m)
        off = {"MED": 0.004, "TENT": -0.001, "AdaBN": -0.008}.get(m, 0)
        ax.text(3.12, v[-1] + off, name, fontsize=7.3, va="center", color=BLUE if m.startswith("CATT") else INK2,
                fontweight="semibold" if m.startswith("CATT") else "normal")
    ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_xlim(-0.2, 3.75); ax.set_ylim(0.68, 0.86)
    ax.set_xlabel("Target prevalence"); ax.set_ylabel("Held-out balanced accuracy (7 settings)")
    hgrid(ax, "y")
    ax.set_title("Best label-free method at every prevalence;\nthe gap is widest for screening", loc="left", pad=22)
    panel(ax, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig4_catt")


# --------------------------------------------------------------------------- Figure 5: pseudo-labeller mechanism
def fig5():
    lab = pd.read_csv(os.path.join(ROOT, "concepts_v6", "v6_labelers.csv"))
    pts = pd.read_csv(os.path.join(ROOT, "concepts_v6", "v6_mechanism_points.csv"))
    concept = {"CBM5", "CBM3", "ST_V2"}
    disp = {"CBM5": "CBM-sd15", "CBM3": "CBM", "ST_V2": "ST_V2", "CNN9": "CNN9", "LogReg9": "LogReg9",
            "ResNet": "ResNet", "CNNalt": "CNNalt"}
    OFF = {"cond_rho": {"CBM5": (-6, -7, "right"), "CBM3": (-6, 6, "right"), "ST_V2": (6, 2, "left"),
                        "LogReg9": (0, 8, "center"), "CNN9": (6, 4, "left"), "CNNalt": (6, 5, "left"), "ResNet": (5, -8, "left")},
           "one_minus": {"CBM5": (7, -6, "left"), "CBM3": (7, 5, "left"), "ST_V2": (-6, 3, "right"),
                         "LogReg9": (-6, 5, "right"), "CNN9": (6, 3, "left"), "CNNalt": (-6, 3, "right"), "ResNet": (6, -3, "left")}}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, xcol, xlab, ttl, letter in [
        (axes[0], "cond_rho", "Within-class correlation with the target CNN", "Labellers that resemble the target fail", "a"),
        (axes[1], "one_minus", "Separation factor 1 − α − β", "Larger separation, smaller gap", "b")]:
        for is_c, c, nm in [(False, ORANGE, "Black-box labeller"), (True, BLUE, "Concept labeller")]:
            q = pts[pts.labeler.isin(concept) == is_c]
            ax.scatter(q[xcol], q.gap, s=14, color=c, alpha=0.35, linewidth=0, zorder=2)
            m = lab[lab.labeler.isin(concept) == is_c]
            ax.scatter(m[xcol], m.gap, s=46, color=c, edgecolor="white", linewidth=1.0, zorder=4, label=nm)
            for _, r in m.iterrows():
                dx, dy, ha = OFF[xcol][r.labeler]
                ax.annotate(disp[r.labeler], (r[xcol], r.gap), xytext=(dx, dy), textcoords="offset points",
                            fontsize=7, color=INK2, ha=ha, va="center", zorder=5)
        ax.set_xlabel(xlab); ax.set_title(ttl, loc="left", pad=14); hgrid(ax, "y")
        ax.set_ylim(-0.005, 0.24)
        panel(ax, letter)
    axes[0].set_ylabel("Gap to oracle (balanced accuracy)")
    axes[0].legend(loc="upper left", frameon=False, handletextpad=0.2)
    for ax, t in [(axes[0], "Spearman ρ = +0.33, p = 0.008 (63 points)"), (axes[1], "Spearman ρ = −0.62, p = 5×10⁻⁸ (63 points)")]:
        ax.text(0, 1.01, t, transform=ax.transAxes, fontsize=7.2, color=MUTED, va="bottom")
    axes[1].text(0.98, 0.98, "small dots: labeller × setting\nlarge dots: mean over 9 settings", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=7, color=MUTED)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig5_labeller_mechanism")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f(); print("ok", f.__name__)
