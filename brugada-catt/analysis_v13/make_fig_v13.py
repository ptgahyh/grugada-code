"""Figure for step 13: Proposition 5 (NEW) vs Propositions 1+3 (OLD), simulation and Brugada.
Run: python analysis_v13/make_fig_v13.py analysis_v13"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = sys.argv[1]
S = pd.read_csv(os.path.join(OUT, "unified_sim.csv"))
B = pd.read_csv(os.path.join(OUT, "brugada_unified.csv")); B = B[B.observed > 1e-4]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
for a, d, title in ((ax[0], S, "A. Simulation (64 cells, 500 runs each)"),
                    (ax[1], B, "B. Brugada, 3 cohorts (189 points)")):
    a.scatter(d.observed, d.pred_old, s=14, c="#bbbbbb", label="Old: Prop. 1 + Prop. 3 (n_eff)")
    a.scatter(d.observed, d.pred_new, s=14, c="#1f5fa8", label="New: Prop. 5 (unified)")
    lo, hi = min(d.observed.min(), d.pred_new.min()) * 0.8, max(d.observed.max(), d.pred_old.max()) * 1.2
    a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel("Observed balanced-accuracy loss of CATT"); a.set_ylabel("Predicted loss")
    r_new = np.exp(np.median(np.log(d.pred_new / d.observed))); r_old = np.exp(np.median(np.log(d.pred_old / d.observed)))
    a.set_title(f"{title}\nmedian pred/obs: new {r_new:.2f}, old {r_old:.2f}", fontsize=9.5)
    a.legend(fontsize=8, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_v13.png"), dpi=170); fig.savefig(os.path.join(OUT, "fig_v13.pdf"))
