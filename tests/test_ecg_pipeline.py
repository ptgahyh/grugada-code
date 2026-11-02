"""
Kiểm thử thuật toán trên ECG GIẢ LẬP có đáp án biết trước (không cần dữ liệu thật).
Chạy: python -m pytest tests/
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ecg_pipeline as ep  # noqa: E402

FS = 100
QRS_S = 0.10      # QRS giả lập dài đúng 100 ms
HR = 75
V1_J_TRUE = 0.20  # ST chênh lên tại điểm J ở V1 (mV)


def synthetic_ecg(hr=HR, seconds=12, fs=FS, ectopic_at=None):
    t = np.arange(int(seconds * fs)) / fs
    sig = np.zeros((len(t), 12))
    gains = np.array([0.8, 1.0, 0.4, -0.7, 0.3, 0.6, -0.9, -0.5, 0.4, 0.9, 1.0, 0.8])
    rr = 60.0 / hr
    for k, onset in enumerate(np.arange(0.3, seconds - 0.8, rr)):
        u = t - onset
        qrs = np.where((u >= 0) & (u <= QRS_S), np.sin(2 * np.pi * u / QRS_S), 0.0)
        t_wave = 0.25 * np.exp(-((u - 0.30) / 0.05) ** 2)
        scale = 3.0 if k == ectopic_at else 1.0
        sig += scale * np.outer(qrs, gains) + np.outer(t_wave, np.abs(gains))
        # ST chênh lên ở V1: tăng mượt trong QRS, giữ đến J rồi giảm dần về 0 sau 200 ms
        st = np.where((u >= 0) & (u < QRS_S), (1 - np.cos(np.pi * u / QRS_S)) / 2, 0.0)
        st = np.where((u >= QRS_S) & (u < QRS_S + 0.2), (1 + np.cos(np.pi * (u - QRS_S) / 0.2)) / 2, st)
        sig[:, ep.LEAD_IDX['V1']] += V1_J_TRUE * st
    return sig


def test_detect_qrs_counts_every_beat():
    sig = ep.bandpass(synthetic_ecg(), FS)
    expected = len(np.arange(0.3, 12 - 0.8, 60.0 / HR))
    assert len(ep.detect_qrs(sig, FS)) == expected


def test_fast_heart_rate_not_missed():
    # Nhịp 140 l/p: code cũ (khoảng cách tối thiểu 0.6 s) sẽ bỏ sót gần nửa số nhịp
    sig = ep.bandpass(synthetic_ecg(hr=140), FS)
    expected = len(np.arange(0.3, 12 - 0.8, 60.0 / 140))
    assert len(ep.detect_qrs(sig, FS)) == expected


def test_qrs_duration_and_j_point():
    sig = ep.bandpass(synthetic_ecg(), FS)
    mb = ep.median_beat(sig, FS)
    feats = ep.features_from_beat(mb['beat'], mb['fid'], FS)
    assert abs(feats['QRS_ms'] - QRS_S * 1000) <= 20
    assert abs(feats['V1_J'] - V1_J_TRUE) <= 0.05
    assert feats['V1_ST80'] < feats['V1_J']  # ST đang giảm dần sau J
    assert feats['V1_Slope'] < 0


def test_heart_rate_and_median_robust_to_outlier_beat():
    clean = ep.median_beat(ep.bandpass(synthetic_ecg(), FS), FS)
    mb = ep.median_beat(ep.bandpass(synthetic_ecg(ectopic_at=4), FS), FS)
    assert abs(60.0 / np.mean(mb['rr']) - HR) < 2
    # Một nhịp biên độ gấp 3 làm lệch trung bình cộng ~15%, nhưng nhịp trung vị gần như không đổi
    assert np.abs(mb['beat'] - clean['beat']).max() < 0.05
