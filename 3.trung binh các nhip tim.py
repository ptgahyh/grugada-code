"""Vẽ nhịp trung bình và các mốc đo (QRS, điểm J, đỉnh/kết thúc T) để kiểm tra bằng mắt.

Cách dùng:
  python "3.trung binh các nhip tim.py"                 # 2 ca mặc định, hiện cửa sổ
  python "3.trung binh các nhip tim.py" 428031 188981   # các ID tùy chọn
  python "3.trung binh các nhip tim.py" --all           # lưu hình mọi ca vào qc_plots/
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from brugada_utils import LEADS, METADATA_PATH, analyze_record, features_from_analysis

QC_LEADS = ['I', 'II', 'aVR', 'V1', 'V2', 'V5']
TRACE = '#2a78d6'
MARK = '#eb6834'
INK = '#0b0b0b'
MUTED = '#52514e'
QRS_BAND = '#e4e3df'


def plot_qc(pid, label=''):
    a = analyze_record(pid)
    if a is None:
        print(f"❌ Không tìm thấy đủ nhịp tim cho ID {pid}")
        return None
    f = features_from_analysis(a)
    fs, r, beat = a['fs'], a['r'], a['beat']
    t = (np.arange(len(beat)) - r) * 1000 / fs

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for ax, name in zip(axes.flat, QC_LEADS):
        x = beat[:, LEADS[name]]
        ax.axvspan(t[a['onset']], t[a['j']], color=QRS_BAND, lw=0)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.plot(t, x, color=TRACE, lw=1.6)
        ax.plot(t[a['j']], x[a['j']], 'o', ms=6, color=INK, mec='white', mew=1.5, zorder=5)
        key = {'II': 't_II', 'V5': 't_V5'}.get(name)
        if key:
            tp, te = a[key]
            if not np.isnan(tp):
                ax.axvline(t[int(tp)], color=MARK, lw=1.2, ls=':')
            if not np.isnan(te):
                ax.axvline(t[int(round(te))], color=MARK, lw=1.2, ls='--')
        ax.set_title(name, loc='left', fontsize=11, color=INK)
        ax.set_xlim(-200, 550)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color(MUTED)
    for ax in axes[1]:
        ax.set_xlabel('Thời gian so với đỉnh R (ms)', color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('mV', color=MUTED, fontsize=9)

    flags = [k for k in ['Type1_Like', 'aVR_Sign', 'S_I_Sign', 'fQRS', 'ER'] if f[k] == 1]
    if not f['quality_ok']:
        flags.append('CHẤT LƯỢNG THẤP')
    fig.suptitle(
        f"ID {pid} {label}\n"
        f"QRS {f['QRS']:.0f} ms · QTc {f['QTc']:.0f} ms · TpTe {f['TpTe']:.0f} ms · "
        f"HR {f['HR']:.0f} · {f['n_used']}/{f['n_beats']} nhịp"
        + (f" · {', '.join(flags)}" if flags else ''),
        x=0.01, ha='left', fontsize=11, color=INK)
    fig.legend(handles=[
        Patch(color=QRS_BAND, label='QRS'),
        Line2D([], [], marker='o', ls='', color=INK, label='Điểm J'),
        Line2D([], [], color=MARK, ls=':', label='Đỉnh T'),
        Line2D([], [], color=MARK, ls='--', label='Kết thúc T'),
    ], loc='upper right', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ids', nargs='*', type=int, help='ID bệnh nhân')
    parser.add_argument('--all', action='store_true', help='lưu hình mọi ca')
    parser.add_argument('--out', default='qc_plots', help='thư mục lưu hình (mặc định qc_plots)')
    args = parser.parse_args()

    if args.all:
        plt.switch_backend('Agg')
        os.makedirs(args.out, exist_ok=True)
        df = pd.read_csv(METADATA_PATH)
        for i, row in enumerate(df.itertuples(index=False), start=1):
            label = f"(brugada={row.brugada}, đột tử={row.sudden_death})"
            fig = plot_qc(row.patient_id, label)
            if fig is not None:
                # Tên file bắt đầu bằng nhãn để dễ lọc: SCD1_B1_428031.png
                name = f"SCD{row.sudden_death}_B{row.brugada}_{row.patient_id}.png"
                fig.savefig(os.path.join(args.out, name), dpi=80)
                plt.close(fig)
            if i % 50 == 0 or i == len(df):
                print(f"✅ Đã vẽ {i}/{len(df)}")
        print(f"Hình được lưu trong thư mục '{args.out}/'")
        return

    cases = [(pid, '') for pid in args.ids] or [
        (428031, "(Brugada, đột tử)"),
        (1142915, "(đột tử, nhãn bình thường)"),
    ]
    for pid, label in cases:
        fig = plot_qc(pid, label)
        if fig is not None and matplotlib.get_backend().lower() == 'agg':
            fig.savefig(f'qc_{pid}.png', dpi=80)
            print(f"Đã lưu qc_{pid}.png")
    plt.show()


if __name__ == '__main__':
    main()
