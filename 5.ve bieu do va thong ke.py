"""
BƯỚC 5: THỐNG KÊ VÀ BIỂU ĐỒ
Ba phép so sánh, mỗi phép trên toàn bộ 14 đặc trưng:
  A. Chẩn đoán : Brugada (brugada >= 1) vs Bình thường
  B. Đột tử    : sudden_death = 1 vs 0 (toàn bộ cohort)
  C. Phân tầng : đột tử vs không, CHỈ trong nhóm Brugada
Với mỗi đặc trưng: median [IQR] từng nhóm, Mann-Whitney U, AUC + KTC 95% bootstrap,
và q-value hiệu chỉnh nhiều phép kiểm định (Benjamini-Hochberg) trong từng phép so sánh.

Vì sao không chỉ báo p-value: nhóm đột tử chỉ có 11 ca (C còn ít hơn). Với cỡ mẫu này p-value rất
không ổn định; AUC kèm khoảng tin cậy cho biết hiệu ứng lớn cỡ nào và chắc chắn tới đâu.
Kết quả B và C chỉ mang tính THĂM DÒ, không đủ để xây mô hình dự báo.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, mannwhitneyu

import ecg_pipeline as ep

DATA_PATH = 'Final_Research_Data.csv'
RESULTS_DIR = 'results'
FIG_DIR = 'figures'
N_BOOT = 2000
SEED = 2505


def auc_with_ci(pos, neg, rng, n_boot=N_BOOT):
    """AUC = P(giá trị nhóm 1 > nhóm 0) = U / (n1*n0). KTC 95% bằng bootstrap phân tầng."""
    def auc(a, b):
        return mannwhitneyu(a, b).statistic / (len(a) * len(b))
    point = auc(pos, neg)
    boots = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def bh_qvalues(p):
    """Hiệu chỉnh Benjamini-Hochberg (kiểm soát tỷ lệ phát hiện sai - FDR)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def fmt_median(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f'{med:.3f} [{q1:.3f}; {q3:.3f}]'


def compare(df, group_col, name, rng):
    rows = []
    for feat in ep.FEATURE_COLUMNS:
        pos = df.loc[df[group_col] == 1, feat].dropna().to_numpy()
        neg = df.loc[df[group_col] == 0, feat].dropna().to_numpy()
        p = mannwhitneyu(pos, neg).pvalue
        auc, lo, hi = auc_with_ci(pos, neg, rng)
        rows.append({'feature': feat, 'mô tả': ep.FEATURE_DESCRIPTIONS[feat],
                     'n1': len(pos), 'n0': len(neg),
                     'median_nhóm1 [IQR]': fmt_median(pos), 'median_nhóm0 [IQR]': fmt_median(neg),
                     'AUC': auc, 'AUC_lo': lo, 'AUC_hi': hi, 'p': p})
    res = pd.DataFrame(rows)
    res['q_BH'] = bh_qvalues(res['p'])
    # Xếp theo độ mạnh hiệu ứng |AUC - 0.5|, không theo p-value
    res = res.reindex((res['AUC'] - 0.5).abs().sort_values(ascending=False).index)
    res.to_csv(os.path.join(RESULTS_DIR, f'stats_{name}.csv'), index=False, float_format='%.4f')
    return res


def plot_top(df, group_col, res, name, labels, top=6):
    feats = res['feature'].head(top).tolist()
    fig, axes = plt.subplots(1, len(feats), figsize=(3.2 * len(feats), 4.5))
    for ax, feat in zip(axes, feats):
        r = res.set_index('feature').loc[feat]
        sns.boxplot(x=group_col, y=feat, data=df, ax=ax, color='white', showfliers=False)
        sns.stripplot(x=group_col, y=feat, data=df, ax=ax, size=3, alpha=0.6, jitter=0.25,
                      hue=group_col, palette='Set1', legend=False)
        ax.set_title(f'{feat}\nAUC {r.AUC:.2f} [{r.AUC_lo:.2f}-{r.AUC_hi:.2f}], q={r.q_BH:.3f}', fontsize=9)
        ax.set_xticks([0, 1], labels)
        ax.set_xlabel('')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'compare_{name}.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(DATA_PATH)
    df = df[df['error'].fillna('') == ''].copy()
    df['brugada_any'] = (df['brugada'] >= 1).astype(int)
    brs = df[df['brugada_any'] == 1]

    analyses = [
        ('A_brugada_vs_normal', df, 'brugada_any', ['Bình thường', 'Brugada']),
        ('B_scd_all', df, 'sudden_death', ['Sống', 'Đột tử']),
        ('C_scd_within_brugada', brs, 'sudden_death', ['Brugada sống', 'Brugada đột tử']),
    ]
    pd.set_option('display.width', 200)
    for name, data, col, labels in analyses:
        res = compare(data, col, name, rng)
        plot_top(data, col, res, name, labels)
        n1 = int((data[col] == 1).sum())
        print(f'\n=== {name}: n1={n1}, n0={len(data) - n1} ===')
        if n1 < 20:
            print(f'⚠️ Chỉ có {n1} ca nhóm 1: kết quả là THĂM DÒ, KTC rất rộng, cần cohort độc lập để xác nhận.')
        print(res[['feature', 'AUC', 'AUC_lo', 'AUC_hi', 'p', 'q_BH']].round(3).to_string(index=False))

    # Kiểu ECG nền (basal_pattern) với đột tử: bảng 2x2 nhỏ -> Fisher exact, không dùng chi bình phương
    tab = pd.crosstab(df['basal_pattern'], df['sudden_death'])
    odds, p = fisher_exact(tab)
    print(f'\n=== basal_pattern x sudden_death (Fisher exact) ===\n{tab}\nOR={odds:.2f}, p={p:.4f}')
    print(f'\n📁 Bảng kết quả: {RESULTS_DIR}/stats_*.csv | 📊 Biểu đồ: {FIG_DIR}/compare_*.png')
