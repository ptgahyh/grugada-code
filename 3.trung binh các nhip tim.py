"""
BƯỚC 3: KIỂM TRA BẰNG MẮT NHỊP TRUNG VỊ VÀ CÁC MỐC ĐO
Vẽ nhịp trung vị của DI, aVR, V1, V2 kèm QRS onset, điểm J, J+80 ms để bạn tự xác minh thuật toán.

Cách chạy:
    python "3.trung binh các nhip tim.py"                 # 2 ca mặc định + toàn bộ ca đột tử
    python "3.trung binh các nhip tim.py" 428031 1142915  # chỉ các ID chỉ định
Hình lưu trong thư mục figures/qc/.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ecg_pipeline as ep

OUT_DIR = os.path.join('figures', 'qc')
PLOT_LEADS = ['I', 'aVR', 'V1', 'V2']


def plot_patient(pid, label=''):
    feats, mb = ep.process_patient(pid)
    if mb is None or 'onset' not in mb:
        print(f'❌ ID {pid}: {feats["error"]}')
        return
    fs, beat = mb['fs'], mb['beat']
    onset, j = mb['onset'], mb['j']
    beat = ep.baseline_correct(beat, onset, fs)
    t_ms = (np.arange(len(beat)) - mb['fid']) / fs * 1000

    fig, axes = plt.subplots(1, len(PLOT_LEADS), figsize=(16, 4), sharey=True)
    for ax, lead in zip(axes, PLOT_LEADS):
        ax.plot(t_ms, beat[:, ep.LEAD_IDX[lead]], color='tab:blue', lw=2)
        ax.axhline(0, color='black', ls='--', alpha=0.4)
        ax.axvline(t_ms[onset], color='tab:green', alpha=0.7, label='QRS onset')
        ax.axvline(t_ms[j], color='tab:red', alpha=0.7, label='Điểm J')
        ax.axvline(t_ms[j] + ep.ST_OFFSET * 1000, color='tab:orange', alpha=0.7, label='J+80 ms')
        ax.set_title(lead)
        ax.set_xlabel('ms (0 = mốc QRS)')
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('mV')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'ID {pid} {label} | QRS={feats["QRS_ms"]:.0f} ms, V1_J={feats["V1_J"]:.2f} mV, '
                 f'V2_J={feats["V2_J"]:.2f} mV, n_beats={feats["n_beats"]}')
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'{pid}.png')
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f'✅ Đã lưu {path}')


if __name__ == '__main__':
    meta = pd.read_csv(ep.METADATA_PATH).set_index('patient_id')
    ids = [int(x) for x in sys.argv[1:]]
    if not ids:
        ids = [428031, 1142915] + meta.index[meta['sudden_death'] == 1].tolist()
    for pid in dict.fromkeys(ids):
        row = meta.loc[pid]
        plot_patient(pid, f'(brugada={row.brugada}, đột tử={row.sudden_death})')
