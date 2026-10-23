"""Screening view of the main model (CNN-V1V3), 7 held-out settings, 3 validation seeds.

Reported per threshold rule (source Youden = SRC, CATT, and target oracle):
  sensitivity, specificity, LR+ and LR-, and PPV / NPV at realistic prevalences (Bayes' rule).
Also sensitivity at 90% and 95% specificity on the target cohort (needs target labels: a property of the
score distribution, comparable to numbers reported in the literature, not a usable threshold rule).
Thresholds are those of the natural-prevalence cohorts; PPV/NPV re-weight their sensitivity/specificity.

Run: python analysis_v10/screening.py harmonized concepts_v5 analysis_v10
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HELD, SETTINGS, cbm_pseudo, load, v5_raw, youden  # noqa: E402

HD, V5, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
X, Y = load(HD)
R5 = v5_raw(V5)
PREV = [0.001, 0.005, 0.01, 0.05]


def se_sp(y, s, t):
    p = s >= t
    return p[y == 1].mean(), (~p[y == 0]).mean()


def se_at_sp(y, s, sp):
    f, t, _ = roc_curve(y, s)
    return t[f <= 1 - sp].max()


rows = []
for k in HELD:
    tgt = SETTINGS[k][1]; yt = Y[tgt]
    for vs in (0, 1, 2):
        r = R5[(k, vs)]; yv = np.array(r["y_val"])
        sv, st = np.array(r["models"]["CNN-V1V3"]["val"]), np.array(r["models"]["CNN-V1V3"]["tgt"])
        th = dict(SRC=youden(yv, sv), CATT=youden(cbm_pseudo(r), st), ORACLE=youden(yt, st))
        for rule, t in th.items():
            se, sp = se_sp(yt, st, t)
            rows.append(dict(k=k, vseed=vs, rule=rule, sens=se, spec=sp))
        rows.append(dict(k=k, vseed=vs, rule="Se at Sp 0.90", sens=se_at_sp(yt, st, 0.90), spec=0.90))
        rows.append(dict(k=k, vseed=vs, rule="Se at Sp 0.95", sens=se_at_sp(yt, st, 0.95), spec=0.95))
df = pd.DataFrame(rows)
per = df.groupby(["rule", "k"])[["sens", "spec"]].mean().reset_index()
summ = per.groupby("rule", sort=False)[["sens", "spec"]].agg(["mean", "min"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
se, sp = summ["sens_mean"], summ["spec_mean"]
summ["LR+"] = se / (1 - sp).clip(lower=1e-6)
summ["LR-"] = (1 - se) / sp
for pi in PREV:
    summ[f"PPV@{pi:.1%}"] = se * pi / (se * pi + (1 - sp) * (1 - pi))
    summ[f"NPV@{pi:.1%}"] = sp * (1 - pi) / (sp * (1 - pi) + (1 - se) * pi)
summ = summ.round(4)
summ.to_csv(os.path.join(OUT, "screening_summary.csv"))
pd.set_option("display.width", 250)
print(summ.to_string())
