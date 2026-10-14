"""Sinh ECG 12 chuyển đạo giả lập (định dạng WFDB, 100 Hz) với hình dạng biết trước."""
import os

import numpy as np
import wfdb

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FS = 100


def _g(t, mu, sd, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sd) ** 2)


def _beat(t, r, lead, brugada, s_wave_i):
    """Một nhịp: P, QRS (q-R-s), T. Đơn vị mV, thời gian giây."""
    p = _g(t, r - 0.17, 0.02, 0.1)
    if lead == 'aVR':
        return p * -1 + _g(t, r, 0.012, -0.8) + _g(t, r + 0.035, 0.012, 0.15) + _g(t, r + 0.27, 0.05, -0.2)
    if lead in ('V1', 'V2'):
        qrs = _g(t, r - 0.01, 0.012, 0.3) + _g(t, r + 0.03, 0.015, -1.0)
        if brugada:  # ST chênh vòm (coved) rồi T âm
            st_t = _g(t, r + 0.08, 0.045, 0.45) + _g(t, r + 0.25, 0.05, -0.3)
        else:
            st_t = _g(t, r + 0.27, 0.05, 0.3)
        return p + qrs + st_t
    s_amp = -0.35 if (lead == 'I' and s_wave_i) else -0.05
    qrs = _g(t, r - 0.035, 0.008, -0.05) + _g(t, r, 0.012, 1.0) + _g(t, r + 0.035, 0.012, s_amp)
    return p + qrs + _g(t, r + 0.27, 0.05, 0.35)


def make_record(base_dir, pid, rr=0.8, brugada=False, s_wave_i=False, rr_pattern=None, seed=0):
    """Ghi bản ghi 12 giây vào base_dir/files/pid/pid.{hea,dat}."""
    rng = np.random.default_rng(seed)
    t = np.arange(12 * FS) / FS
    rrs = rr_pattern or [rr]
    r_times, r, k = [], 0.5, 0
    while r < 11.5:
        r_times.append(r)
        r += rrs[k % len(rrs)]
        k += 1
    sig = np.zeros((len(t), 12))
    for i, lead in enumerate(LEAD_NAMES):
        for rt in r_times:
            sig[:, i] += _beat(t, rt, lead, brugada, s_wave_i)
    sig += rng.normal(0, 0.005, sig.shape) + 0.05 * np.sin(2 * np.pi * 0.15 * t)[:, None]
    d = os.path.join(base_dir, 'files', str(pid))
    os.makedirs(d, exist_ok=True)
    wfdb.wrsamp(str(pid), fs=FS, units=['mV'] * 12, sig_name=LEAD_NAMES, p_signal=sig,
                fmt=['16'] * 12, write_dir=d)
