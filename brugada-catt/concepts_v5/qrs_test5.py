"""(Bước 5) Kiểm chứng QRS_rel so với QRS_w, không cần bác sĩ.

  Q1  bất biến afin từng chuyển đạo (tín hiệu thô so với z-score), ECG giả lập
  Q2  gradient hữu hạn
  Q3  giãn QRS KHU TRÚ ở V1–V3 (mô phỏng chậm dẫn truyền thất phải) so với giãn TOÀN BỘ chuyển đạo
      (mô phỏng khác dải thông máy đo hoặc rối loạn dẫn truyền lan tỏa):
      khái niệm tốt phải tăng theo giãn khu trú và gần như không đổi khi giãn toàn bộ
  Q4  nhiễu loạn "thiết bị" trên ECG thật âm tính: lọc thông thấp 20/25/30 Hz, làm trơn 3 điểm.
      Đo SMD giữa trước và sau nhiễu loạn
  Chế độ: abs = QRS_w (v3), rel = QRS_rel (chọn trước), lp15 = QRS_w15 (lọc 15 Hz, bị loại),
       sd15 = QRS_sd15 (thời lượng RMS sau lọc 15 Hz, được chọn)
  Q5  dịch chuyển giữa các miền thật ở nhóm âm tính (SMD) và AUROC theo từng miền

Chạy: python qrs_test5.py <harmonized> <output>
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.signal import butter, filtfilt, resample_poly
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from concepts5 import ANCHOR, BrugadaConcepts  # noqa: E402

HD, OUT = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
MODES = ["abs", "rel", "lp15", "sd15"]
EXT = {w: BrugadaConcepts(learn=False, qrs=w).double() for w in MODES}
rep = {}


def concept(x, which):
    with torch.no_grad():
        return EXT[which](torch.tensor(np.asarray(x, dtype=np.float64)))[:, -1].numpy()


def zscore(b):
    return (b - b.mean(-1, keepdims=True)) / (b.std(-1, keepdims=True) + 1e-8)


def align(beat):
    e = np.abs(np.diff(beat, axis=1)).sum(0)
    return np.roll(beat, ANCHOR - int(np.argmax(e)), axis=1)


# ------------------------------------------------------------ ECG giả lập có thể giãn QRS
FS_HI = 1000
t = (np.arange(750) - 290) / FS_HI * 1000.0


def g(c, w, a):
    return a * np.exp(-0.5 * ((t - c) / w) ** 2)


def beat(k_local=1.0, k_global=1.0, gain=None):
    """k_local: hệ số giãn QRS của V1–V3; k_global: hệ số giãn QRS của mọi chuyển đạo."""
    def qrs_pre(k):
        return g(-8 * k, 7 * k, 0.35) + g(18 * k, 9 * k, -1.2) + g(60 * k - 20, 9 * k, 0.15)
    def qrs_lat(k, a):
        return g(0, 8 * k, a) + g(22 * k, 8 * k, -0.25 * a)
    P, T = g(-170, 22, 0.1), g(250, 45, 0.25)
    limb = [P + qrs_lat(k_global, a) + T for a in (1.0, 1.2, 0.4)]
    pre = [P + qrs_pre(k_global * k_local) * f + g(250, 45, 0.3) for f in (1.0, 1.1, 0.6)]
    lat = [P + qrs_lat(k_global, a) + T for a in (1.2, 1.4, 1.1)]
    hi = np.stack(limb + pre + lat)
    if gain is not None:
        hi = hi * gain[:, None] + rng.normal(0, 1, 9)[:, None]
    lo = resample_poly(hi, 1, 10, axis=1)[:, :75] + rng.normal(0, 0.01, (9, 75))
    return align(lo)


# Q1
Xr = np.stack([beat(rng.uniform(1, 1.4), rng.uniform(0.8, 1.3), rng.uniform(0.3, 3, 9)) for _ in range(100)])
Xz = zscore(Xr)
rep["Q1_affine_max_diff"] = {w: float(np.abs(concept(Xr, w) - concept(Xz, w)).max()) for w in MODES}

# Q2
rep["Q2_grad_finite"] = {}
for w in MODES:
    m = BrugadaConcepts(learn=True, qrs=w).double()
    x = torch.tensor(Xz[:16]); m(x).sum().backward()
    rep["Q2_grad_finite"][w] = bool(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

# Q3
ks = np.linspace(1.0, 1.5, 11)
loc = np.stack([zscore(beat(k_local=k)) for k in ks])
glb = np.stack([zscore(beat(k_global=k)) for k in ks])
q3 = {}
for w in MODES:
    a, b = concept(loc, w), concept(glb, w)
    sd_ref = np.std(concept(Xz, w))
    q3[w] = dict(local_spearman=round(float(spearmanr(ks, a)[0]), 3), global_spearman=round(float(spearmanr(ks, b)[0]), 3),
                 local_range_sd=round(float((a[-1] - a[0]) / sd_ref), 2), global_range_sd=round(float((b[-1] - b[0]) / sd_ref), 2),
                 local=a.round(4).tolist(), glob=b.round(4).tolist())
rep["Q3_widening"] = q3
rep["Q3_k"] = ks.round(3).tolist()

# ------------------------------------------------------------ dữ liệu thật
X = {d: np.load(os.path.join(HD, f"{d}.npz"))["X"].astype(np.float64) for d in ["huca", "claris", "mortara"]}
Y = {d: np.load(os.path.join(HD, f"{d}.npz"))["y"].astype(int) for d in X}
C = {d: {w: concept(X[d], w) for w in MODES} for d in X}


def smd(a, b):
    return float((b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2))


# Q4: nhiễu loạn thiết bị trên nhóm âm tính
q4 = {}
for name, fn in [("lowpass30", lambda z: filtfilt(*butter(4, 30 / 50), z, axis=-1)),
                 ("lowpass25", lambda z: filtfilt(*butter(4, 25 / 50), z, axis=-1)),
                 ("lowpass20", lambda z: filtfilt(*butter(4, 20 / 50), z, axis=-1)),
                 ("smooth3", lambda z: np.apply_along_axis(lambda v: np.convolve(v, np.ones(3) / 3, "same"), -1, z))]:
    q4[name] = {}
    for d in X:
        x0 = X[d][Y[d] == 0]
        x1 = zscore(fn(x0))
        q4[name][d] = {w: round(smd(concept(x0, w), concept(x1, w)), 3) for w in MODES}
rep["Q4_device_perturbation_SMD"] = q4

# Q5: dịch chuyển giữa miền và AUROC theo miền
q5 = {"shift_neg_SMD": {}, "auroc": {}}
for a, b in [("claris", "mortara"), ("huca", "claris"), ("huca", "mortara")]:
    q5["shift_neg_SMD"][f"{a}->{b}"] = {w: round(smd(C[a][w][Y[a] == 0], C[b][w][Y[b] == 0]), 3) for w in MODES}
for d in X:
    q5["auroc"][d] = {w: round(float(roc_auc_score(Y[d], C[d][w])), 3) for w in MODES}
rep["Q5_real"] = q5

json.dump(rep, open(os.path.join(OUT, "qrs_test5.json"), "w"), indent=1)
for k, v in rep.items():
    if k == "Q3_widening":
        for w, r in v.items():
            print(k, w, {kk: r[kk] for kk in ["local_spearman", "global_spearman", "local_range_sd", "global_range_sd"]})
    elif k != "Q3_k":
        print(k, v)
