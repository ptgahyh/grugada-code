"""Accurate Chernoff samples (argmax of two-sided BM minus s^2), cached to chernoff_samples.npy.
Fine grid h = 2.5e-4 on [-3, 3] (P(|Z| > 3) < 1e-9), 40,000 samples, vectorised. Theory (Groeneboom & Wellner
2001): Var Z = 0.26355, SD = 0.5134. Run: python analysis_v13/chernoff.py analysis_v13"""
import os
import sys

import numpy as np

OUT = sys.argv[1]
rng = np.random.default_rng(2001)
H, L, M, B = 2.5e-4, 3.0, 40000, 200
s = np.arange(0, L + H, H)
out = []
for _ in range(M // B):
    w = np.cumsum(rng.standard_normal((2, B, len(s) - 1)), axis=2) * np.sqrt(H)
    w = np.concatenate([np.zeros((2, B, 1)), w], axis=2) - s ** 2
    i1, i2 = w[0].argmax(1), w[1].argmax(1)
    m1, m2 = w[0].max(1), w[1].max(1)
    out.append(np.where(m1 >= m2, s[i1], -s[i2]))
Z = np.concatenate(out)
np.save(os.path.join(OUT, "chernoff_samples.npy"), Z.astype(np.float32))
print("n", len(Z), "var", round(Z.var(), 4), "+-", round(Z.var() * np.sqrt(2.2 / len(Z)), 4), "(theory 0.26355)")
