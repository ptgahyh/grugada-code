"""Bước 20: ngưỡng Youden làm trơn (xem PLAN_vi.md). f phải được sắp tăng dần cho `raw_thr`; `smooth_thr` không cần."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

G = 1024


def bandwidth(f, pl, mult=1.0):
    n1 = pl.sum(); nmin = min(n1, len(f) - n1)
    q75, q25 = np.percentile(f, [75, 25]); a = min(f.std(), (q75 - q25) / 1.34)
    return 0.9 * a * nmin ** (-0.2) * mult


def _grid(f, h):
    lo, hi = f.min() - 3 * h, f.max() + 3 * h; d = (hi - lo) / G
    return lo, d, np.minimum(((f - lo) / d).astype(int), G - 1)


def _argmax_smooth(d1, d0, lo, d, h):
    """d1, d0: khối lượng theo ô (đã chuẩn hóa tổng 1) của hai lớp nhãn giả. J_h = làm trơn Gauss của (S1 − S0)."""
    s = np.cumsum((d1 - d0)[::-1])[::-1]                       # S1 − S0 tại mép trái mỗi ô
    j = gaussian_filter1d(s, h / d, mode="nearest", truncate=5.0)
    k = int(np.argmax(j)); off = 0.0
    if 0 < k < G - 1:
        den = j[k - 1] - 2 * j[k] + j[k + 1]
        if den < 0:
            off = 0.5 * (j[k - 1] - j[k + 1]) / den
    return lo + (k + off) * d


def smooth_thr(f, pl, mult=1.0, h=None):
    if h is None:
        h = bandwidth(f, pl, mult)
    lo, d, idx = _grid(f, h); n1 = pl.sum()
    d1 = np.bincount(idx, weights=pl, minlength=G) / n1; d0 = np.bincount(idx, weights=1 - pl, minlength=G) / (len(f) - n1)
    return _argmax_smooth(d1, d0, lo, d, h)


def smooth_boot_sd(f, pl, rng, B=200, mult=1.0):
    """SD bootstrap thường của t̂_h; h và lưới giữ cố định theo mẫu gốc."""
    h = bandwidth(f, pl, mult); lo, d, idx = _grid(f, h); n = len(f); th = []
    for _ in range(B):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float); w1 = w * pl; w0 = w - w1
        if w1.sum() == 0 or w0.sum() == 0:
            continue
        th.append(_argmax_smooth(np.bincount(idx, weights=w1, minlength=G) / w1.sum(),
                                 np.bincount(idx, weights=w0, minlength=G) / w0.sum(), lo, d, h))
    return float(np.std(th)) if len(th) >= 20 else np.nan


def raw_thr(f, pl):
    """Ngưỡng Youden thường trên mẫu đã sắp tăng dần (quy tắc f ≥ t)."""
    n1 = pl.sum(); n0 = len(f) - n1
    j = np.cumsum(pl[::-1])[::-1] / n1 - np.cumsum((1 - pl)[::-1])[::-1] / n0
    return f[int(np.argmax(j))]
