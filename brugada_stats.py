"""Các hàm thống kê cho mẫu nhỏ, ít biến cố (11 ca đột tử)."""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, fisher_exact, mannwhitneyu, norm


def auc(neg, pos):
    """AUC = P(giá trị nhóm biến cố > nhóm không biến cố). 0.5 = không phân biệt."""
    neg, pos = np.sort(np.asarray(neg, float)), np.asarray(pos, float)
    below = np.searchsorted(neg, pos, side='left')
    ties = np.searchsorted(neg, pos, side='right') - below
    return (below + 0.5 * ties).sum() / (len(pos) * len(neg))


def bootstrap_auc_ci(neg, pos, n_boot=2000, seed=0, level=0.95):
    """Khoảng tin cậy AUC bằng bootstrap phân tầng (lấy mẫu lại trong từng nhóm)."""
    rng = np.random.default_rng(seed)
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    stats = [auc(rng.choice(neg, len(neg)), rng.choice(pos, len(pos))) for _ in range(n_boot)]
    a = (1 - level) / 2
    return tuple(np.quantile(stats, [a, 1 - a]))


def firth_logit(X, y, max_iter=100, tol=1e-8):
    """Hồi quy logistic Firth (giảm sai lệch khi ít biến cố hoặc tách biệt hoàn toàn).

    X phải chứa cột hằng số. Trả về (hệ số, sai số chuẩn, log-likelihood có phạt).
    """
    X, y = np.asarray(X, float), np.asarray(y, float)

    def penalized_loglik(b):
        eta = X @ b
        w = expit(eta) * (1 - expit(eta))
        logdet = np.linalg.slogdet(X.T @ (X * w[:, None]))[1]
        return np.sum(y * eta - np.logaddexp(0, eta)) + 0.5 * logdet

    beta = np.zeros(X.shape[1])
    ll = penalized_loglik(beta)
    for _ in range(max_iter):
        pi = expit(X @ beta)
        w = pi * (1 - pi)
        inv = np.linalg.inv(X.T @ (X * w[:, None]))
        h = np.einsum('ij,jk,ik->i', X, inv, X) * w
        step = inv @ (X.T @ (y - pi + h * (0.5 - pi)))
        for _ in range(30):  # giảm nửa bước nếu likelihood không tăng
            ll_new = penalized_loglik(beta + step)
            if ll_new >= ll - 1e-10:
                break
            step /= 2
        beta, ll = beta + step, ll_new
        if np.max(np.abs(step)) < tol:
            break
    pi = expit(X @ beta)
    w = pi * (1 - pi)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * w[:, None]))))
    return beta, se, ll


def firth_univariate(x, y):
    """Firth 1 biến: OR (mỗi 1 đơn vị x), CI 95% Wald, p theo kiểm định tỉ số likelihood có phạt."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ones = np.ones_like(x)
    beta, se, ll_full = firth_logit(np.column_stack([ones, x]), y)
    _, _, ll_null = firth_logit(ones[:, None], y)
    p = chi2.sf(max(2 * (ll_full - ll_null), 0), 1)
    z = norm.ppf(0.975)
    return np.exp(beta[1]), np.exp(beta[1] - z * se[1]), np.exp(beta[1] + z * se[1]), p


def bh_fdr(pvalues):
    """Giá trị q theo Benjamini-Hochberg (bỏ qua NaN)."""
    p = pd.Series(pvalues, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    q = (valid * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
    return q.reindex(p.index).to_numpy()


def compare_groups(df, outcome, continuous, binary, n_boot=2000):
    """So sánh nhóm outcome=1 với outcome=0 cho từng đặc trưng.

    Liên tục: trung vị [IQR], Mann-Whitney, AUC (CI bootstrap), Firth OR mỗi 1 SD.
    Nhị phân: n/N (%), Fisher, Firth OR.
    """
    rows = []
    for col in continuous + binary:
        sub = df[[col, outcome]].dropna()
        neg = sub.loc[sub[outcome] == 0, col]
        pos = sub.loc[sub[outcome] == 1, col]
        row = {'feature': col, 'n_0': len(neg), 'n_1': len(pos)}
        if col in continuous:
            q0, q1 = neg.quantile([.25, .5, .75]), pos.quantile([.25, .5, .75])
            row['group_0'] = f"{q0[.5]:.3g} [{q0[.25]:.3g}; {q0[.75]:.3g}]"
            row['group_1'] = f"{q1[.5]:.3g} [{q1[.25]:.3g}; {q1[.75]:.3g}]"
            row['p_test'] = mannwhitneyu(pos, neg).pvalue
            row['auc'] = auc(neg, pos)
            row['auc_lo'], row['auc_hi'] = bootstrap_auc_ci(neg, pos, n_boot)
            sd = sub[col].std()
            x = (sub[col] - sub[col].mean()) / sd if sd > 0 else None
        else:
            a, b = int(pos.sum()), int(neg.sum())
            row['group_0'] = f"{b}/{len(neg)} ({100 * b / max(len(neg), 1):.0f}%)"
            row['group_1'] = f"{a}/{len(pos)} ({100 * a / max(len(pos), 1):.0f}%)"
            row['p_test'] = fisher_exact([[a, len(pos) - a], [b, len(neg) - b]]).pvalue
            x = sub[col] if 0 < sub[col].sum() < len(sub) else None
        if x is not None:
            row['or'], row['or_lo'], row['or_hi'], row['p_firth'] = firth_univariate(x, sub[outcome])
        rows.append(row)
    out = pd.DataFrame(rows)
    out['q_fdr'] = bh_fdr(out['p_test'])
    return out
