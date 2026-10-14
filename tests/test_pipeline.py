"""Kiểm thử pipeline bằng ECG giả lập. Chạy: python -m pytest tests/"""
import os
import sys

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import brugada_utils as bu  # noqa: E402
from brugada_stats import auc, bh_fdr, firth_logit  # noqa: E402
from synthetic_ecg import make_record  # noqa: E402


@pytest.fixture(scope='module')
def records(tmp_path_factory):
    base = tmp_path_factory.mktemp('ecg')
    make_record(base, 1)                                   # bình thường, 75 lần/phút
    make_record(base, 2, brugada=True)                     # dạng Brugada type 1 ở V1-V2
    make_record(base, 3, s_wave_i=True)                    # sóng S sâu ở DI
    make_record(base, 4, rr_pattern=[0.55, 1.1])           # nhịp không đều
    make_record(base, 5, rr=0.5)                           # nhịp nhanh 120 lần/phút
    old = bu.FILES_DIR
    bu.FILES_DIR = os.path.join(base, 'files')
    yield {pid: bu.extract_features(pid) for pid in [1, 2, 3, 4, 5]}
    bu.FILES_DIR = old


def test_normal_record(records):
    f = records[1]
    assert f['HR'] == pytest.approx(75, abs=1)
    assert 70 <= f['QRS'] <= 120
    assert 330 <= f['QTc'] <= 480
    assert 30 <= f['TpTe'] <= 120
    assert f['Type1_Like'] == 0
    assert f['S_I_Sign'] == 0
    assert f['quality_ok'] == 1
    assert f['n_used'] == f['n_beats']


def test_brugada_type1_detected(records):
    normal, brug = records[1], records[2]
    assert brug['Type1_Like'] == 1
    assert brug['V1_J'] > normal['V1_J'] + 0.1
    assert brug['V1_Slope'] < 0
    assert brug['V1_T_Amp'] < -0.1


def test_s_wave_lead_i(records):
    f = records[3]
    assert f['S_I_Sign'] == 1
    assert f['S_I_Amp'] >= 0.2


def test_irregular_rhythm_flagged(records):
    f = records[4]
    assert f['RR_CV'] > bu.MAX_RR_CV
    assert f['quality_ok'] == 0


def test_tachycardia_does_not_break_measurements(records):
    f = records[5]
    assert f['HR'] == pytest.approx(120, abs=2)
    assert 70 <= f['QRS'] <= 120
    assert f['Type1_Like'] == 0


def test_missing_record_returns_nan(records):
    f = bu.extract_features(999)
    assert set(f) == set(bu.FEATURE_COLUMNS)
    assert all(np.isnan(v) for v in f.values())


def test_auc_matches_mann_whitney():
    rng = np.random.default_rng(1)
    neg = rng.integers(0, 20, 300).astype(float)  # có giá trị trùng
    pos = rng.integers(3, 25, 11).astype(float)
    expected = mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))
    assert auc(neg, pos) == pytest.approx(expected)


def test_bh_fdr():
    q = bh_fdr([0.01, 0.04, 0.03, np.nan, 0.5])
    assert q[0] == pytest.approx(0.04)
    assert q[1] == pytest.approx(0.04 * 4 / 3)  # p=0.04 xếp hạng 3/4
    assert q[2] == pytest.approx(0.04 * 4 / 3)  # p=0.03: 0.06, lấy min với hạng sau
    assert np.isnan(q[3])
    assert q[4] == pytest.approx(0.5)


def test_firth_close_to_ml_on_large_sample():
    rng = np.random.default_rng(0)
    x = rng.normal(size=3000)
    y = (rng.uniform(size=3000) < expit(-1 + 0.8 * x)).astype(float)
    X = np.column_stack([np.ones_like(x), x])
    beta_firth, _, _ = firth_logit(X, y)
    nll = lambda b: -np.sum(y * (X @ b) - np.logaddexp(0, X @ b))
    beta_ml = minimize(nll, np.zeros(2)).x
    assert beta_firth == pytest.approx(beta_ml, abs=0.02)


def test_firth_finite_under_separation():
    x = np.array([0., 0, 0, 0, 1, 1, 1, 1])
    y = np.array([0., 0, 0, 0, 1, 1, 1, 1])  # tách biệt hoàn toàn: ML không hội tụ
    beta, se, _ = firth_logit(np.column_stack([np.ones_like(x), x]), y)
    assert np.all(np.isfinite(beta)) and np.all(np.isfinite(se))
    assert beta[1] > 0
