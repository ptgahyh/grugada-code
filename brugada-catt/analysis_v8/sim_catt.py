"""(Bước 8) Mô phỏng kiểm chứng lý thuyết CATT (Mục 2.8) trong điều kiện kiểm soát được.

Lô đích: y ~ Bernoulli(π). Điểm mô hình đích f và bộ gán nhãn g theo lớp là Gauss hai chiều:
  f | y ~ N(μ_f,y, σ_f,y²),  g | y ~ N(μ_g,y, 1),  corr(f, g | y) = ρ   (ρ = phụ thuộc có điều kiện)
Phương sai không bằng nhau (σ_f,1 = 1.6, σ_f,0 = 1) để ngưỡng Youden thật KHÔNG nằm ở điểm giữa, giống điểm CNN bị
nén dưới dịch chuyển. Bộ gán nhãn g có độ tách d_g (AUROC = Φ(d_g/√2)) và ngưỡng của g bị đặt sai một lượng c
(mô phỏng ngưỡng nguồn của g không còn đúng ở đích) -> thay đổi α, β.
Đo: khoảng cách balanced accuracy giữa CATT và ORACLE (ngưỡng Youden thật), trung bình 300 lần lặp.
Kiểm tra ba dự đoán:
  (1) ρ = 0: khoảng cách → 0 khi n tăng, BẤT KỂ g lệch ngưỡng (c) bao nhiêu (miễn α + β < 1).
  (2) khoảng cách tăng theo ρ.
  (3) một g chính xác hơn nhưng phụ thuộc hơn có thể tệ hơn một g yếu hơn nhưng độc lập.
Chạy: python sim_catt.py <out>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT = sys.argv[1]
rng = np.random.default_rng(7)
MU_F = (0.0, 1.6); SD_F = (1.0, 1.6)


def youden_thr(y, s):
    o = np.argsort(-s); ss = s[o]; yy = y[o]
    tp = np.cumsum(yy) / yy.sum(); fp = np.cumsum(1 - yy) / (1 - yy).sum()
    return ss[np.argmax(tp - fp)]


def true_bacc(t):
    return 0.5 * (norm.sf(t, MU_F[1], SD_F[1]) + norm.cdf(t, MU_F[0], SD_F[0]))


grid = np.linspace(-4, 6, 4001)
t_star = grid[np.argmax([true_bacc(t) for t in grid])]
best = true_bacc(t_star)


def sample(n, pi, rho, dg, c):
    y = (rng.random(n) < pi).astype(int)
    z1 = rng.standard_normal(n); z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    f = np.where(y == 1, MU_F[1] + SD_F[1] * z1, MU_F[0] + SD_F[0] * z1)
    g = np.where(y == 1, dg + z2, z2)
    pl = (g >= dg / 2 + c).astype(int)
    return y, f, pl


rows = []
for n in (100, 300, 1000, 3000):
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9):
        for dg in (1.0, 1.5, 2.5):
            for c in (0.0, 0.75):
                gaps, ab = [], []
                for _ in range(300):
                    y, f, pl = sample(n, 0.2, rho, dg, c)
                    if pl.min() == pl.max():
                        continue
                    t = youden_thr(pl, f)
                    gaps.append(best - true_bacc(t))
                    ab.append(pl[y == 0].mean() + (1 - pl[y == 1]).mean())
                rows.append(dict(n=n, rho=rho, g_auroc=round(float(norm.cdf(dg / np.sqrt(2))), 3), c=c,
                                 alpha_plus_beta=float(np.mean(ab)), gap_mean=float(np.mean(gaps)),
                                 gap_p90=float(np.percentile(gaps, 90))))
D = pd.DataFrame(rows)
D.round(4).to_csv(os.path.join(OUT, "sim_catt.csv"), index=False)
summary = {
    "oracle_bacc": round(float(best), 4), "true_youden_threshold": round(float(t_star), 3),
    "rho0_gap_by_n_and_miscalibration": D[(D.rho == 0) & (D.g_auroc == D.g_auroc.min())].pivot_table(index="n", columns="c", values="gap_mean").round(4).to_dict(),
    "gap_by_rho_n1000": D[(D.n == 1000) & (D.c == 0.75)].pivot_table(index="rho", columns="g_auroc", values="gap_mean").round(4).to_dict(),
}
a = D[(D.n == 1000) & (D.c == 0.75)]
weak_indep = a[(a.rho == 0.2) & (a.g_auroc == a.g_auroc.min())].gap_mean.iloc[0]
strong_dep = a[(a.rho == 0.9) & (a.g_auroc == a.g_auroc.max())].gap_mean.iloc[0]
summary["weak_independent_vs_strong_dependent_gap"] = dict(weak_auc=float(a.g_auroc.min()), weak_rho=0.2, gap_weak=round(float(weak_indep), 4),
                                                           strong_auc=float(a.g_auroc.max()), strong_rho=0.9, gap_strong=round(float(strong_dep), 4))
json.dump(summary, open(os.path.join(OUT, "sim_catt_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))

# ------------------------------------------------------------------ kịch bản B: lỗi CHUNG do một nhóm con đích
# Một phần q của nhóm âm tính đích mang một "đặc điểm site" làm điểm f tăng thêm δ (ví dụ một kiểu hình hay nhiễu
# riêng của máy). Bộ gán nhãn g độc lập (κ = 0) không bị ảnh hưởng; g phụ thuộc (κ = 1) cũng tăng điểm ở chính nhóm đó.
# Đây là dạng vi phạm độc lập có điều kiện mà dữ liệu thật gợi ý: hai mạng giống nhau mắc cùng một kiểu lỗi.
def true_bacc_B(t, q, delta):
    spec = (1 - q) * norm.cdf(t, MU_F[0], SD_F[0]) + q * norm.cdf(t, MU_F[0] + delta, SD_F[0])
    return 0.5 * (norm.sf(t, MU_F[1], SD_F[1]) + spec)


rowsB = []
for q in (0.0, 0.2, 0.4):
    delta = 1.5
    tb = grid[np.argmax([true_bacc_B(t, q, delta) for t in grid])]; bb = true_bacc_B(tb, q, delta)
    for kappa in (0.0, 0.5, 1.0):
        for dg in (1.5, 2.5):
            gaps, ab = [], []
            for _ in range(300):
                n = 1000
                y = (rng.random(n) < 0.2).astype(int)
                art = (y == 0) & (rng.random(n) < q)
                f = np.where(y == 1, MU_F[1] + SD_F[1] * rng.standard_normal(n), MU_F[0] + SD_F[0] * rng.standard_normal(n)) + delta * art
                g = dg * y + rng.standard_normal(n) + kappa * dg * art
                pl = (g >= dg / 2).astype(int)
                t = youden_thr(pl, f)
                gaps.append(bb - true_bacc_B(t, q, delta)); ab.append(pl[y == 0].mean() + (1 - pl[y == 1]).mean())
            rowsB.append(dict(q=q, kappa=kappa, g_auroc_clean=round(float(norm.cdf(dg / np.sqrt(2))), 3),
                              alpha_plus_beta=float(np.mean(ab)), gap_mean=float(np.mean(gaps))))
DB = pd.DataFrame(rowsB)
DB.round(4).to_csv(os.path.join(OUT, "sim_catt_shared_error.csv"), index=False)
print(DB.round(4).to_string(index=False))

# ------------------------------------------------------------------ kịch bản C: ngưỡng của g lệch nặng (gần như gán cùng một lớp)
# Đây là điều xảy ra với bộ gán nhãn hộp đen trong dữ liệu thật: ngưỡng nguồn của nó lệch CÙNG CHIỀU với f,
# nên ở đích gần như mọi ECG bị gán dương (α ≈ 0.8–0.9). Thay đổi c để quét 1 − α − β, với ρ = 0 và ρ = 0.9.
rowsC = []
for n in (356, 1154):
    for rho in (0.0, 0.9):
        for c in (0.0, -0.75, -1.25, -1.75, -2.25):
            gaps, sep = [], []
            for _ in range(300):
                y, f, pl = sample(n, 0.2, rho, 2.5, c)
                if pl.min() == pl.max():
                    gaps.append(np.nan); continue
                t = youden_thr(pl, f)
                gaps.append(best - true_bacc(t)); sep.append(1 - pl[y == 0].mean() - (1 - pl[y == 1]).mean())
            rowsC.append(dict(n=n, rho=rho, c=c, one_minus_alpha_beta=float(np.mean(sep)), gap_mean=float(np.nanmean(gaps)),
                              gap_p90=float(np.nanpercentile(gaps, 90))))
DC = pd.DataFrame(rowsC)
DC.round(4).to_csv(os.path.join(OUT, "sim_catt_degenerate.csv"), index=False)
print(DC.round(4).to_string(index=False))
