"""Thống kê và biểu đồ.

1. Kiểm chứng phép đo: Brugada (brugada=1) so với người khỏe (brugada=0)
2. Phân tích chính: Đột tử so với Sống, toàn bộ bệnh nhân
3. Phân nhóm: chỉ bệnh nhân Brugada (brugada=1 hoặc 2)
4. Độ nhạy: chỉ các ca đạt chất lượng tín hiệu (quality_ok=1)
5. Hiệu chỉnh: Firth 2 biến (đặc trưng + chẩn đoán Brugada) cho các đặc trưng có q < 0.05 ở (2)

Với chỉ 11 ca đột tử, mọi kết quả mang tính thăm dò. Giá trị q (FDR
Benjamini-Hochberg) hiệu chỉnh cho việc kiểm định nhiều đặc trưng cùng lúc.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brugada_stats import compare_groups, firth_logit
from brugada_utils import BINARY_FEATURES, CONTINUOUS_FEATURES

LABELS = {
    'V1_J': 'Chênh J V1 (mV)', 'V1_Slope': 'Độ dốc ST V1 (mV/s)', 'V1_T_Amp': 'ST-T thấp nhất V1 (mV)',
    'V2_J': 'Chênh J V2 (mV)', 'V2_Slope': 'Độ dốc ST V2 (mV/s)', 'V2_T_Amp': 'ST-T thấp nhất V2 (mV)',
    'QRS': 'Độ rộng QRS (ms)', 'aVR_R': 'Sóng R aVR (mV)', 'S_I_Amp': 'Biên độ S ở DI (mV)',
    'S_I_Dur': 'Thời gian S ở DI (ms)', 'fQRS_Count': 'Số đỉnh/đáy QRS V1-V3', 'TpTe': 'Tpeak-Tend V5 (ms)',
    'QTc': 'QTc Bazett (ms)', 'HR': 'Nhịp tim (lần/phút)',
    'Type1_Like': 'Dạng Brugada type 1', 'aVR_Sign': 'Dấu aVR (R ≥ 0.3 mV)',
    'S_I_Sign': 'Sóng S ở DI (≥0.1 mV/≥40 ms)', 'fQRS': 'QRS phân mảnh', 'ER': 'Tái cực sớm',
}
GROUP_0 = '#2a78d6'
GROUP_1 = '#eb6834'
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e4e3df'
FIG_DIR = 'figures'


def print_table(res, title, name0, name1):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(f"{'Đặc trưng':32s} {name0:>24s} {name1:>24s} {'AUC [95% CI]':>18s} {'p':>7s} {'q':>6s}")
    for r in res.itertuples():
        auc = f"{r.auc:.2f} [{r.auc_lo:.2f}-{r.auc_hi:.2f}]" if not pd.isna(r.auc) else ''
        mark = ' *' if r.q_fdr < 0.05 else ''
        print(f"{LABELS[r.feature]:32s} {r.group_0:>24s} {r.group_1:>24s} {auc:>18s} "
              f"{r.p_test:7.3g} {r.q_fdr:6.2g}{mark}")
    print("(*) q < 0.05 sau hiệu chỉnh FDR. Liên tục: trung vị [IQR], Mann-Whitney. Nhị phân: n/N (%), Fisher.")


def style(ax):
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(MUTED)


def plot_auc_forest(res, path, title):
    r = res.dropna(subset=['auc']).sort_values('auc')
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(r) + 1.4))
    y = np.arange(len(r))
    ax.axvline(0.5, color=MUTED, lw=1, ls='--')
    ax.hlines(y, r.auc_lo, r.auc_hi, color=GROUP_0, lw=2)
    ax.plot(r.auc, y, 'o', ms=8, color=GROUP_0, mec='white', mew=2)
    for yi, row in zip(y, r.itertuples()):
        ax.text(1.01, yi, f"{row.auc:.2f}", va='center', fontsize=9, color=INK,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [LABELS[f] for f in r.feature], color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel('AUC (95% CI bootstrap) — > 0.5: giá trị cao hơn ở nhóm đột tử', color=MUTED)
    ax.set_title(title, loc='left', color=INK, fontsize=12)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_boxes(df, res, outcome, path, names):
    top = res.dropna(subset=['auc']).assign(d=lambda r: (r.auc - 0.5).abs()).nlargest(6, 'd').feature
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    rng = np.random.default_rng(0)
    for ax, col in zip(axes.flat, top):
        groups = [df.loc[df[outcome] == g, col].dropna() for g in (0, 1)]
        bp = ax.boxplot(groups, widths=0.5, showfliers=False, patch_artist=True,
                        medianprops={'color': INK, 'lw': 2}, whiskerprops={'color': MUTED},
                        capprops={'color': MUTED})
        for patch, c in zip(bp['boxes'], (GROUP_0, GROUP_1)):
            patch.set(facecolor=c, alpha=0.25, edgecolor=c)
        for i, (vals, c) in enumerate(zip(groups, (GROUP_0, GROUP_1)), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=14 if i == 1 else 36,
                       color=c, alpha=0.45 if i == 1 else 0.9, edgecolor='white', lw=0.6, zorder=3)
        a = res.set_index('feature').loc[col]
        ax.set_title(f"{LABELS[col]}\nAUC {a.auc:.2f}, p = {a.p_test:.2g}", loc='left', fontsize=10, color=INK)
        ax.set_xticks([1, 2], [f"{names[0]}\n(n={len(groups[0])})", f"{names[1]}\n(n={len(groups[1])})"])
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def adjusted_for_brugada(df, features):
    """OR (mỗi 1 SD) của đặc trưng sau khi hiệu chỉnh theo chẩn đoán Brugada (Firth, 2 biến)."""
    print(f"\n{'=' * 100}\nHiệu chỉnh theo chẩn đoán Brugada (Firth 2 biến; 11 biến cố nên chỉ mang tính tham khảo)\n{'=' * 100}")
    rows = []
    for col in features:
        sub = df[[col, 'brugada', 'sudden_death']].dropna()
        x = (sub[col] - sub[col].mean()) / sub[col].std()
        X = np.column_stack([np.ones(len(sub)), x, (sub['brugada'] > 0).astype(float)])
        beta, se, _ = firth_logit(X, sub['sudden_death'])
        lo, hi = np.exp(beta[1] - 1.96 * se[1]), np.exp(beta[1] + 1.96 * se[1])
        print(f"{LABELS[col]:32s} OR mỗi 1 SD = {np.exp(beta[1]):.2f} [95% CI {lo:.2f}-{hi:.2f}]")
        rows.append({'analysis': 'hieu_chinh_brugada', 'feature': col, 'n_0': int((sub.sudden_death == 0).sum()),
                     'n_1': int(sub.sudden_death.sum()), 'or': np.exp(beta[1]), 'or_lo': lo, 'or_hi': hi})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv('Final_Research_Data.csv')
    n_fail = df['QRS'].isna().sum()
    df = df.dropna(subset=['QRS'])
    print(f"Số ca: {len(df)} (loại {n_fail} ca không trích được đặc trưng)")
    print(pd.crosstab(df['brugada'], df['sudden_death'], margins=True))

    analyses = [
        ('kiem_chung', 'Kiểm chứng phép đo: Brugada (1) so với người khỏe (0)',
         df[df['brugada'] < 2].assign(_y=lambda d: d['brugada']), ('Khỏe', 'Brugada')),
        ('chinh', 'Phân tích chính: Đột tử so với Sống — toàn bộ bệnh nhân',
         df.assign(_y=df['sudden_death']), ('Sống', 'Đột tử')),
        ('nhom_brugada', 'Phân nhóm: Đột tử so với Sống — chỉ bệnh nhân Brugada (brugada = 1, 2)',
         df[df['brugada'] > 0].assign(_y=lambda d: d['sudden_death']), ('Sống', 'Đột tử')),
        ('do_nhay', 'Độ nhạy: Đột tử so với Sống — chỉ ca đạt chất lượng (quality_ok = 1)',
         df[df['quality_ok'] == 1].assign(_y=lambda d: d['sudden_death']), ('Sống', 'Đột tử')),
    ]

    os.makedirs(FIG_DIR, exist_ok=True)
    all_res = []
    for key, title, data, names in analyses:
        res = compare_groups(data, '_y', CONTINUOUS_FEATURES, BINARY_FEATURES)
        print_table(res, f"{title}  [n = {len(data)}, nhóm 1 = {int(data['_y'].sum())}]", *names)
        all_res.append(res.assign(analysis=key))
        if key in ('kiem_chung', 'chinh'):
            plot_auc_forest(res, os.path.join(FIG_DIR, f'auc_{key}.png'), title)
        if key == 'chinh':
            plot_boxes(data, res, '_y', os.path.join(FIG_DIR, 'so_sanh_dac_trung.png'), names)

    main_res = all_res[1]
    sig = main_res.loc[(main_res.q_fdr < 0.05) & main_res.feature.isin(CONTINUOUS_FEATURES), 'feature'].tolist()
    if sig:
        all_res.append(adjusted_for_brugada(df, sig))

    out = pd.concat(all_res)
    cols = ['analysis', 'feature', 'n_0', 'n_1', 'group_0', 'group_1', 'p_test', 'q_fdr',
            'auc', 'auc_lo', 'auc_hi', 'or', 'or_lo', 'or_hi', 'p_firth']
    out[cols].round(4).to_csv('ket_qua_thong_ke.csv', index=False)
    print("\nOR Firth (mỗi 1 SD với biến liên tục) có trong ket_qua_thong_ke.csv.")
    print(f"Đã lưu: ket_qua_thong_ke.csv, {FIG_DIR}/auc_kiem_chung.png, {FIG_DIR}/auc_chinh.png, "
          f"{FIG_DIR}/so_sanh_dac_trung.png")


if __name__ == '__main__':
    main()
