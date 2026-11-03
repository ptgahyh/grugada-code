"""Hàm dùng chung cho bước 15 (giống analysis_v14/analyze_rbbb.py, tách ra để import được)."""
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve

SDZ = float(np.sqrt(0.26355))      # SD của phân phối Chernoff


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def youden(y, s):
    if y.min() == y.max():
        return None
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.minimum(thr, s.max() + 1e-9)
    return thr[np.argmax(tpr - fpr)]


def kde(x, m):
    return gaussian_kde(x, bw_method=lambda k: m * k.scotts_factor())


def pseudo_terms(f, pl, m, g):
    """(t̃, Ṽ, Ã) từ KDE băng thông Scott × m của f trong từng lớp nhãn giả."""
    q = pl.mean(); dg = g[1] - g[0]
    d1, d0 = kde(f[pl == 1], m)(g), kde(f[pl == 0], m)(g)
    i = int(np.argmax(np.cumsum(d0 - d1))); i = min(max(i, 2), len(g) - 3)
    return g[i], d1[i] / q + d0[i] / (1 - q), abs(np.gradient(d0 - d1, dg)[i])


def sd_formula(V, A, n, N):
    """Định lý 5a: SD(t̂) ≈ SD(Z)·(4Ṽ(1 − n/N)/(Ã²n))^{1/3}."""
    return SDZ * (4 * V * (1 - n / N) / (A ** 2 * n)) ** (1 / 3)


def n_needed(V, A, eps, N):
    n_inf = 4 * SDZ ** 3 * V / (A ** 2 * eps ** 3)
    return n_inf / (1 + n_inf / N)


def sd_subsample(f, pl, rng, B=100, frac=0.5):
    """Ước lượng không nhãn bằng mẫu con (m-out-of-n, Politis–Romano) trên MỘT mẫu cỡ n (f, nhãn giả pl).

    Lấy B mẫu con cỡ b = frac·n không hoàn lại, tính SD của t̂_b. Theo Định lý 5a áp cho "quần thể" là mẫu cỡ n,
    SD(t̂_b) ≈ c·((1 − b/n)/b)^{1/3}; trả về c (để nhân với ((1 − n'/N)/n')^{1/3} cho cỡ mẫu n' bất kỳ).
    Không cần ước lượng mật độ hay đạo hàm, nên tự phản ánh đuôi dày khi nhóm nhãn giả dương nhỏ.
    """
    n = len(f); b = int(round(frac * n)); th = []
    for _ in range(B):
        idx = rng.choice(n, b, replace=False)
        if pl[idx].min() != pl[idx].max():
            th.append(youden(pl[idx], f[idx]))
    if len(th) < B // 2:
        return None
    return float(np.std(th)) / ((1 - b / n) / b) ** (1 / 3)


def sd_from_c(c, n, N):
    return c * ((1 - n / N) / n) ** (1 / 3)


def n_from_c(c, eps, N):
    """Giải c·((1 − n/N)/n)^{1/3} = ε theo n."""
    k = (c / eps) ** 3          # (1 − n/N)/n = 1/k  ->  n = k/(1 + k/N)
    return k / (1 + k / N)
