"""Bước 5A: hiệu chỉnh ngưỡng ở miền đích KHÔNG dùng nhãn miền đích.

Vấn đề (bước 4): AUROC ngoài miền tốt nhưng ngưỡng chọn trên validation nguồn bị lệch có hệ thống
(Melo -> HUCA: điểm bị đẩy lên, độ đặc hiệu 0.06–0.38; HUCA -> Melo: điểm bị kéo xuống, độ nhạy thấp).
Đây là dịch chuyển của ĐIỂM (covariate shift), không phải dịch chuyển tỉ lệ bệnh (label shift): khi chỉ có
label shift, mật độ điểm theo từng lớp không đổi nên ngưỡng Youden (tỉ số hợp lý ~ 1) không đổi; vì vậy các
phương pháp chỉnh theo tỉ lệ bệnh (EM của Saerens 2002, BBSE) không sửa được lỗi này.

Phương pháp (mọi phương pháp chỉ dùng nhãn ở validation NGUỒN và điểm KHÔNG nhãn ở miền đích):
  SRC     ngưỡng Youden trên validation nguồn (như bước 4)
  MED     chuẩn hóa bền: đưa trung vị/IQR của điểm đích về trung vị/IQR của điểm validation nguồn
          (tương đương "Ada" ở mức điểm; phụ thuộc tỉ lệ bệnh vì trung vị trộn cả hai lớp)
  LSM     hỗn hợp dịch chuyển vị trí (đề xuất BAN ĐẦU): trong thang logit, mật độ điểm đích
              g(s) = (1 − π) N(s; μ0 + δ, σ0) + π N(s; μ1 + δ, σ1),   μk, σk từ validation nguồn
          δ, π ước lượng bằng hợp lý cực đại (lưới δ + EM cho π, π ≤ 0.6); ngưỡng đích = ngưỡng nguồn + δ.
          Trên tập phát triển, LSM THẤT BẠI với CNN vì điểm CNN ngoài miền bị NÉN (khoảng cách hai lớp co
          còn 1/5), không chỉ dịch. Điểm của mô hình khái niệm thì chủ yếu dịch, nên LSM được giữ lại để
          hiệu chỉnh chính mô hình khái niệm (bước trong CATT).
  CATT    (đề xuất CUỐI, "concept-anchored threshold transfer"): nhãn giả từ mô hình khái niệm CBM5-Sign
          tại ngưỡng nguồn của nó; ngưỡng của mô hình đích = Youden theo nhãn giả trên miền đích.
          Cơ sở lý thuyết: ngưỡng tối ưu balanced accuracy (Youden) KHÔNG ĐỔI dưới nhiễu nhãn phụ thuộc lớp
          (Menon và cs., ICML 2015): J_nhiễu(t) = (1 − α − β)·J(t), với α, β là tỉ lệ lật nhãn của nhãn giả,
          nếu lỗi của nhãn giả độc lập có điều kiện (theo lớp thật) với điểm của mô hình đích và α + β < 1.
          Vì vậy ngưỡng lệch của chính mô hình khái niệm KHÔNG cần sửa; nó chỉ làm (1 − α − β) nhỏ đi.
  CATT-LSM  như CATT nhưng ngưỡng CBM5-Sign được dời trước bằng LSM. Trên tập phát triển KHÔNG tốt hơn CATT
          (đúng như lý thuyết trên), nên chỉ báo cáo làm biến thể.
  ORACLE  ngưỡng Youden dùng nhãn đích (cận trên, không dùng được trong thực tế)

Quy trình chống "chọn theo kết quả": phương pháp được phát triển CHỈ trên 2 thiết lập HUCA -> Claris và
HUCA -> Mortara (setting 0, 1). Kết quả báo cáo riêng cho 7 thiết lập còn lại (held-out).

Thử tải tỉ lệ bệnh: lấy mẫu con nhóm dương tính ở miền đích để tỉ lệ bệnh = 5%, 10%, 20% (10 lần rút),
cùng với tỉ lệ tự nhiên.

Chạy: python recal_v5.py <thư mục chứa v5_raw_*.jsonl> <harmonized>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

OD, HD = sys.argv[1], sys.argv[2]
Y = {d: np.load(os.path.join(HD, f"{d}.npz"))["y"].astype(int) for d in ["huca", "claris", "mortara"]}
ROWS = [json.loads(l) for f in sorted(glob.glob(os.path.join(OD, "v5_raw_*.jsonl"))) for l in open(f)]
MODELS = ["CNN-V1V3", "CBM3-Sign", "CBM5-Sign", "Hybrid-S3", "Hybrid-S5", "Hybrid-A5"]   # MODELS[0] phải có mặt
PREVS = [0.05, 0.10, 0.20, None]
NDRAW = 10
PI_MAX = 0.6


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.minimum(thr, s.max() + 1e-9)
    return thr[np.argmax(tpr - fpr)]


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def gauss_params(v):
    """vị trí/thang bền (trung vị, IQR/1.349), có sàn cho tập rất nhỏ."""
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return med, max((q3 - q1) / 1.349, 0.25)


def lsm_fit(s, sv, yv, affine=False, iters=40):
    """trả về (δ, a, π̂). EM cho π được vector hóa trên toàn lưới δ (và a)."""
    m0, s0 = gauss_params(sv[yv == 0]); m1, s1 = gauss_params(sv[yv == 1])
    deltas = np.arange(-8, 8.001, 0.1)
    scales = np.exp(np.linspace(np.log(0.5), np.log(2.0), 9)) if affine else np.array([1.0])
    best = (-np.inf, 0.0, 1.0, 0.1)
    for a in scales:
        f0 = npdf(s[None, :], a * m0 + deltas[:, None], a * s0) + 1e-300
        f1 = npdf(s[None, :], a * m1 + deltas[:, None], a * s1) + 1e-300
        pi = np.full((len(deltas), 1), 0.2)
        for _ in range(iters):
            r = pi * f1 / (pi * f1 + (1 - pi) * f0)
            pi = np.clip(r.mean(1, keepdims=True), 0.005, PI_MAX)
        ll = np.log(pi * f1 + (1 - pi) * f0).sum(1)
        j = int(np.argmax(ll))
        if ll[j] > best[0]:
            best = (ll[j], float(deltas[j]), float(a), float(pi[j, 0]))
    return best[1], best[2], best[3]


def bal(y, p):
    se = float(p[y == 1].mean()); sp = float((~p)[y == 0].mean())
    return se, sp, (se + sp) / 2


out = []
rng = np.random.default_rng(0)
for r in ROWS:
    yv = np.array(r["y_val"]); yt_full = Y[r["test"]]
    cbm_v = logit(r["models"]["CBM5-Sign"]["val"]); cbm_t_full = logit(r["models"]["CBM5-Sign"]["tgt"])
    pl_thr = youden(yv, cbm_v)
    for prev in PREVS:
        draws = [None] if prev is None else range(NDRAW)
        for k in draws:
            if prev is None:
                idx = np.arange(len(yt_full))
            else:
                neg = np.where(yt_full == 0)[0]; pos = np.where(yt_full == 1)[0]
                npos = max(1, int(round(prev / (1 - prev) * len(neg))))
                if npos > len(pos):          # không đủ dương tính -> bớt âm tính
                    npos = len(pos); neg = rng.choice(neg, int(round(npos * (1 - prev) / prev)), replace=False)
                idx = np.concatenate([neg, rng.choice(pos, npos, replace=False)])
            yt = yt_full[idx]
            for m in MODELS:
                sv = logit(r["models"][m]["val"]); st = logit(np.array(r["models"][m]["tgt"])[idx])
                t_src = youden(yv, sv)
                thr = {"SRC": t_src}
                q1v, mv, q3v = np.percentile(sv, [25, 50, 75]); q1t, mt, q3t = np.percentile(st, [25, 50, 75])
                thr["MED"] = mt + (t_src - mv) * (q3t - q1t) / max(q3v - q1v, 1e-6)
                d, _, pi_hat = lsm_fit(st, sv, yv)
                thr["LSM"] = t_src + d
                ct = cbm_t_full[idx]
                pl0 = (ct >= pl_thr).astype(int)
                thr["CATT"] = youden(pl0, st) if 0 < pl0.mean() < 1 else t_src
                if m == MODELS[0]:          # LSM trên điểm khái niệm: chỉ cần tính một lần cho mỗi lần rút
                    dc, _, pic = lsm_fit(ct, cbm_v, yv)
                plc = (ct >= pl_thr + dc).astype(int)
                thr["CATT-LSM"] = youden(plc, st) if 0 < plc.mean() < 1 else t_src
                thr["ORACLE"] = youden(yt, st)
                for meth, t in thr.items():
                    se, sp, ba = bal(yt, st >= t)
                    out.append(dict(setting=r["setting"], train=r["train"], test=r["test"], vseed=r["vseed"], prev=prev or "tự nhiên",
                                    true_prev=round(float(yt.mean()), 3), draw=-1 if k is None else k, model=m, method=meth,
                                    sens=se, spec=sp, bacc=ba,
                                    pi_hat=pi_hat if meth == "LSM" else (pic if meth == "CATT-LSM" else np.nan)))
    print("xong", r["train"], "->", r["test"], "vseed", r["vseed"], flush=True)
df = pd.DataFrame(out)
df.to_csv(os.path.join(OD, "recal_raw.csv"), index=False)
print(len(df), "dòng")
