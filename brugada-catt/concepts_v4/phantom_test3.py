"""Kiểm chứng lớp khái niệm bằng ECG giả lập (không cần bác sĩ).

Sinh nhịp ở 1000 Hz với hình học biết trước (độ chênh J, dạng ST vòm / yên ngựa / bình thường,
biên độ r'), hạ xuống 100 Hz, thêm nhiễu, đổi biên độ ngẫu nhiên, rồi đi qua đúng pipeline
căn chỉnh + z-score như dữ liệu thật. So sánh khái niệm trích được với giá trị thật.

Các kiểm tra:
  T1  bất biến afin: khái niệm trên tín hiệu mV thô == trên tín hiệu z-score (từng chuyển đạo)
  T2  gradient: đạo hàm theo tham số mốc hữu hạn và khớp sai phân hữu hạn
  T3  khôi phục: tương quan J_elev / ST80 trích được với giá trị thật; sai số vị trí J
  T4  hình dạng: phân biệt vòm vs yên ngựa bằng Corrado, curv (AUROC)
  T5  bền vững: nhiễu trắng + trôi nền + lệch căn chỉnh ±20 ms
"""
import json
import sys

import numpy as np
import torch
from scipy.signal import resample_poly
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
from concepts3 import ANCHOR, CONCEPT_NAMES, BrugadaConcepts  # noqa: E402

rng = np.random.default_rng(0)
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- tiền xử lý giống harmonize.py
def align(beat, anchor=ANCHOR):
    e = np.abs(np.diff(beat, axis=1)).sum(0)
    shift = anchor - int(np.argmax(e))
    if shift == 0:
        return beat
    out = np.empty_like(beat)
    if shift > 0:
        out[:, shift:] = beat[:, :-shift]; out[:, :shift] = beat[:, :1]
    else:
        out[:, :shift] = beat[:, -shift:]; out[:, shift:] = beat[:, -1:]
    return out


def zscore(b):
    return (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-8)


# ---------------------------------------------------------------- bộ sinh nhịp
FS_HI = 1000
t = (np.arange(750) - 290) / FS_HI * 1000.0          # ms, QRS (đỉnh R) tại 0 ms


def gauss(c, w, a):
    return a * np.exp(-0.5 * ((t - c) / w) ** 2)


def precordial(shape, e, rprime, J_ms):
    """shape: 'normal' | 'coved' | 'saddle'; e: độ chênh J (mV); rprime: biên độ r' (mV)."""
    y = gauss(-170, 22, 0.12)                                         # P
    y += gauss(-8, 7, 0.35) + gauss(18, 9, -1.2)                       # r nhỏ + S sâu (V1/V2)
    y += gauss(J_ms - 20, 9, rprime)                                   # r' cuối QRS
    s = 1 / (1 + np.exp(-(t - J_ms) / 3.0))                            # bước mềm tại J
    u = np.clip((t - J_ms) / 120.0, 0, 1)
    if shape == "coved":
        st = e * np.cos(np.pi / 2 * u)                                 # vòm, lõm xuống, đi xuống
        Tw = gauss(J_ms + 190, 45, -0.35)                              # T âm
    elif shape == "saddle":
        st = e * (1 - 0.6 * np.sin(np.pi * np.clip(u / 0.5, 0, 1)))   # yên ngựa: lõm xuống đáy ở J+30 ms rồi đi lên
        Tw = gauss(J_ms + 190, 45, 0.35)                               # T dương
    else:
        st = e * (1 - u)
        Tw = gauss(J_ms + 190, 45, 0.30)
    y += s * st * (t <= J_ms + 120) + Tw
    return y


def limb(amp):
    return gauss(-170, 22, 0.1) + gauss(0, 8, amp) + gauss(22, 8, -0.25 * amp) + gauss(250, 45, 0.25)


def make_beat(shape, e, rprime, gain, noise, wander, jitter_ms=0):
    J_ms = 80.0
    leads = [limb(1.0), limb(1.2), limb(0.4)]
    pre = [precordial(shape, e * f, rprime * f, J_ms) for f in (1.0, 1.1, 0.6)]   # V1, V2, V3
    lat = [limb(a) for a in (1.2, 1.4, 1.1)]                                       # V4..V6 (thô)
    hi = np.stack(leads + pre + lat) * gain[:, None]
    hi += rng.normal(0, 1, 9)[:, None] * 0.5                                      # lệch DC tùy ý
    hi += wander * np.sin(2 * np.pi * 0.3 * t / 1000 + rng.uniform(0, 6))[None]   # trôi nền
    if jitter_ms:
        hi = np.roll(hi, int(jitter_ms), axis=1)
    lo = resample_poly(hi, 1, 10, axis=1)                                         # 1000 -> 100 Hz
    lo = lo[:, :75] + rng.normal(0, noise, (9, 75))
    truth = {}
    for name, li, f in [("V1", 3, 1.0), ("V2", 4, 1.1), ("V3", 5, 0.6)]:
        y = precordial(shape, e * f, rprime * f, J_ms)
        qrs = y[(t > -60) & (t < 80)]
        pp = qrs.max() - qrs.min()
        # điểm cất cánh thật: cực đại của y trong [J-20, J+40] ms (đo ở 1000 Hz)
        win = np.where((t >= J_ms - 20) & (t <= J_ms + 40))[0]
        j = win[np.argmax(y[win])]; j80 = j + 80
        truth[f"J_elev_{name}"] = y[j] / pp
        truth[f"ST80_{name}"] = y[j80] / pp
    return lo, truth


def batch(n, shapes, noise=0.02, wander=0.05, jitter=False, raw=False):
    X, T, S = [], [], []
    for _ in range(n):
        sh = shapes[rng.integers(len(shapes))]
        e = {"normal": rng.uniform(0.0, 0.08), "coved": rng.uniform(0.15, 0.6), "saddle": rng.uniform(0.1, 0.4)}[sh]
        rp = rng.uniform(0.05, 0.5) if sh != "normal" else rng.uniform(0.0, 0.1)
        gain = rng.uniform(0.3, 3.0, 9)
        lo, tr = make_beat(sh, e, rp, gain, noise, wander, jitter_ms=rng.integers(-20, 21) if jitter else 0)
        b = align(lo)
        X.append(b if raw else zscore(b)); T.append(tr); S.append(sh)
    return np.stack(X), T, np.array(S)


ext = BrugadaConcepts(learn=True).double()
report = {}

# T1: bất biến afin
Xr, _, _ = batch(200, ["normal", "coved", "saddle"], raw=True)
Xz = np.stack([zscore(b) for b in Xr])
with torch.no_grad():
    Cr = ext(torch.tensor(Xr)); Cz = ext(torch.tensor(Xz))
diff = (Cr - Cz).abs().max(0).values
qrs_idx = CONCEPT_NAMES.index("QRS_w")
report["T1_max_abs_diff_ratio_concepts"] = float(diff[:qrs_idx].max())
report["T1_QRS_w_max_diff_ms"] = float(diff[qrs_idx] * 100)

# T2: gradient
x = torch.tensor(Xz[:32])
C = ext(x).sum()
C.backward()
grads = {n: float(p.grad) for n, p in ext.named_parameters()}
report["T2_grads_finite"] = all(np.isfinite(list(grads.values())))
eps = 1e-5
with torch.no_grad():
    p = ext.p_theta; base_v = ext(x).sum().item(); p += eps; up = ext(x).sum().item(); p -= eps
report["T2_fd_vs_autograd_theta"] = [round((up - base_v) / eps, 4), round(grads["p_theta"], 4)]

# T3 + T4 + T5
for cond, kw in {"sach": dict(noise=0.005, wander=0.0, jitter=False),
                 "nhieu": dict(noise=0.03, wander=0.15, jitter=False),
                 "nhieu+lech": dict(noise=0.03, wander=0.15, jitter=True)}.items():
    X, T, S = batch(600, ["normal", "coved", "saddle"], **kw)
    with torch.no_grad():
        Cc, aux = ext(torch.tensor(X), return_aux=True)
    Cc = Cc.numpy()
    r = {}
    for k in ["J_elev_V1", "J_elev_V2", "ST80_V1", "ST80_V2"]:
        est = Cc[:, CONCEPT_NAMES.index(k)]; tru = np.array([d[k] for d in T])
        r[k + "_spearman"] = round(float(spearmanr(est, tru)[0]), 3)
    m = S != "normal"
    yc = (S[m] == "coved").astype(int)
    for k in ["Corrado_V1", "Corrado_V2", "curv60_V1", "curv60_V2", "beta_rel_V1", "beta_rel_V2", "T_V2"]:
        sgn = -1 if (k.startswith("curv") or k.startswith("T_")) else 1   # vòm: curv âm, T âm, beta rộng
        r[k + "_AUC_coved_vs_saddle"] = round(roc_auc_score(yc, sgn * Cc[m, CONCEPT_NAMES.index(k)]), 3)
    yb = (S != "normal").astype(int)
    r["J_elev_V2_AUC_abnormal_vs_normal"] = round(roc_auc_score(yb, Cc[:, CONCEPT_NAMES.index("J_elev_V2")]), 3)
    # vị trí J: thật = 80 ms sau đỉnh R; đỉnh năng lượng sau căn chỉnh ở ANCHOR (~ đỉnh R ± vài ms)
    r["J_pos_ms_after_anchor_median"] = round(float(np.median((aux["J"].numpy() - ANCHOR) * 10)), 1)
    r["J_pos_ms_IQR"] = [round(float(v), 1) for v in np.percentile((aux["J"].numpy() - ANCHOR) * 10, [25, 75])]
    report["T3-5_" + cond] = r

print(json.dumps(report, indent=1, ensure_ascii=False))
json.dump(report, open("phantom_report.json", "w"), indent=1, ensure_ascii=False)

# T6: đo dưới mức một mẫu (nội suy + soft-argmax) so với đo "cứng" trên lưới nguyên 10 ms
X, T, S = batch(600, ["normal", "coved", "saddle"], noise=0.03, wander=0.15, jitter=True)
with torch.no_grad():
    Cc, aux = ext(torch.tensor(X), return_aux=True)
J = aux["J"].numpy(); base = aux["base"].numpy()
t6 = {}
for name, li in [("V1", 3), ("V2", 4)]:
    tru = np.array([d[f"J_elev_{name}"] for d in T])
    soft = Cc[:, CONCEPT_NAMES.index(f"J_elev_{name}")].numpy()
    hard = []
    for i in range(len(X)):
        x = X[i, li]; j0 = int(round(J[i]))
        w = x[j0 - 1:j0 + 6] - base[i, li]                       # cùng cửa sổ, chỉ lấy mẫu nguyên
        seg = x[ANCHOR - 6:ANCHOR + 9]
        hard.append(w.max() / (seg.max() - seg.min()))
    t6[f"J_elev_{name}_spearman_soft"] = round(float(spearmanr(soft, tru)[0]), 3)
    t6[f"J_elev_{name}_spearman_hard"] = round(float(spearmanr(np.array(hard), tru)[0]), 3)
report["T6_subsample_vs_hard"] = t6
print(json.dumps(t6, indent=1))
json.dump(report, open("phantom_report.json", "w"), indent=1, ensure_ascii=False)
