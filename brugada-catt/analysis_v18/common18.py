"""Bước 18: hàm dùng chung. Mô hình mô phỏng Gauss của bài (như analysis_v13/unified_sim.py) và các ước lượng SD(t̂).

Quy ước: quy tắc quyết định f ≥ t; t̂ = điểm mẫu f_(k) làm lớn nhất J(t) = S1(t) − S0(t), với S_k(t) là tỉ lệ nhóm
nhãn giả k có f ≥ t. Mọi phép lấy lại mẫu được viết dưới dạng trọng số trên mẫu đã sắp xếp, nên mỗi lần chỉ tốn O(n).
"""
import numpy as np
from scipy.stats import gaussian_kde

PI = 0.2
MU = np.array([0.0, 1.6]); SD = np.array([1.0, 1.6]); NU = np.array([0.0, 2.5]); MID = 1.25
SDZ = float(np.sqrt(0.26355))
CONFIGS = [(0.0, 0.0), (0.0, -1.25), (0.6, 0.0), (0.6, -1.25)]     # (rho, độ dời ngưỡng mỏ neo)


def draw(rng, n, rho, shift):
    """n bản ghi độc lập từ quần thể: trả về f (đã sắp tăng dần) và nhãn giả pl tương ứng."""
    y = (rng.random(n) < PI).astype(int)
    z1 = rng.standard_normal(n); z2 = rng.standard_normal(n)
    f = MU[y] + SD[y] * z1
    g = NU[y] + rho * z1 + np.sqrt(1 - rho ** 2) * z2
    pl = (g >= MID + shift).astype(float)
    o = np.argsort(f)
    return f[o], pl[o]


def curves(W, pl):
    """W: (B, n) trọng số không âm trên mẫu đã sắp. Trả về J (B, n) và mặt nạ hàng hợp lệ (cả hai lớp có trọng số)."""
    w1 = W * pl; w0 = W * (1 - pl)
    s1 = w1.sum(1); s0 = w0.sum(1); ok = (s1 > 0) & (s0 > 0)
    c1 = np.cumsum(w1[:, ::-1], 1)[:, ::-1] / np.where(ok, s1, 1)[:, None]
    c0 = np.cumsum(w0[:, ::-1], 1)[:, ::-1] / np.where(ok, s0, 1)[:, None]
    return c1 - c0, ok


def that(f, pl):
    J, ok = curves(np.ones((1, len(f))), pl)
    return (f[int(np.argmax(J[0]))], J[0]) if ok[0] else (None, None)


def subsample_masks(rng, B, n, b):
    """B mẫu con cỡ b không hoàn lại, dạng mặt nạ 0/1 (B, n)."""
    k = np.argpartition(rng.random((B, n)), b - 1, axis=1)[:, :b]
    M = np.zeros((B, n)); np.put_along_axis(M, k, 1.0, 1)
    return M


def boot_counts(rng, B, n):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)


def sd_of(th):
    return float(np.std(th)) if len(th) >= 20 else np.nan


def kde_terms(f, pl, m=1.0):
    """(Ṽ, Ã) từ KDE Scott × m của f trong từng lớp nhãn giả (như common15.pseudo_terms)."""
    q = pl.mean(); g = np.linspace(f.min() - 1, f.max() + 1, 1201); dg = g[1] - g[0]
    k = lambda x: gaussian_kde(x, bw_method=lambda s: m * s.scotts_factor())
    d1, d0 = k(f[pl == 1])(g), k(f[pl == 0])(g)
    i = int(np.argmax(np.cumsum(d0 - d1))); i = min(max(i, 2), len(g) - 3)
    return d1[i] / q + d0[i] / (1 - q), abs(np.gradient(d0 - d1, dg)[i])


def estimators(f, pl, rng, eps_pows=(1 / 3, 1 / 4, 0.4)):
    """Mọi ước lượng không nhãn của SD(t̂_n) (N = ∞) từ MỘT mẫu (f đã sắp, pl). Trả về dict."""
    n = len(f); out = {}
    n1 = pl.sum()
    if n1 < 3 or n - n1 < 3:
        return None
    t0, J0 = that(f, pl)
    # E0 plug-in
    try:
        V, A = kde_terms(f, pl)
        out["E0_plugin"] = SDZ * (4 * V / (A ** 2 * n)) ** (1 / 3)
    except Exception:
        V = A = np.nan; out["E0_plugin"] = np.nan
    # E1 F1 nửa mẫu (B = 100, b = n/2; hệ số quy đổi = 1 khi N = ∞)
    b = int(round(0.5 * n)); J, ok = curves(subsample_masks(rng, 100, n, b), pl)
    th = f[np.argmax(J, 1)][ok]
    out["E1_half"] = float(np.std(th)) if ok.sum() >= 50 else np.nan
    # E2 mẫu con b = n^(2/3)
    b = int(round(n ** (2 / 3))); J, ok = curves(subsample_masks(rng, 200, n, b), pl)
    out["E2_sub23"] = sd_of(f[np.argmax(J, 1)][ok]) * (b / n) ** (1 / 3)
    # bootstrap có hoàn lại: dùng chung cho E3, E4, E5
    Js, ok = curves(boot_counts(rng, 200, n), pl); Js = Js[ok]; D = Js - J0[None]
    out["E5_boot"] = sd_of(f[np.argmax(Js, 1)])
    out["E3_reshape"] = sd_of(f[np.argmax(D - 0.5 * A * (f - t0)[None] ** 2, 1)]) if np.isfinite(A) else np.nan
    for p in eps_pows:
        eps = n ** (-p)
        th = f[np.argmax(J0[None] + eps * np.sqrt(n) * D, 1)]
        out[f"E4_num_{p:.2f}"] = sd_of(th) * eps ** (-2 / 3) * n ** (-1 / 3)
    return out
