"""Kiểm chứng điểm J của lớp khái niệm trên LUDB (bác sĩ tim mạch đánh dấu ranh giới sóng).

Chuẩn tham chiếu: điểm kết thúc QRS (ký hiệu ')' ngay sau 'N') trong file chú thích .v2 của LUDB.
Mô hình KHÔNG thấy chú thích: tín hiệu đi qua đúng quy trình harmonize.py (lọc 0.5–40 Hz, 100 Hz,
dò đỉnh bằng năng lượng đạo hàm, nhịp trung vị 75 mẫu, căn đỉnh năng lượng về mẫu 28, z-score),
rồi lớp khái niệm khả vi trả về điểm J (chỉ số mẫu thực trong cửa sổ nhịp đã căn).

Đổi điểm J về thời gian tuyệt đối của từng nhịp k:
    t_J,k = (p_k − PRE + J − shift) × 10 ms
với p_k là đỉnh dò được (100 Hz), shift là độ dịch khi căn. Sai số của bản ghi = trung vị theo nhịp
của (t_J,k − t_offset,k), chỉ tính các nhịp ghép được với một 'N' có chú thích trong ±100 ms.

So sánh 6 biến thể:
  v2-init     năng lượng 9 chuyển đạo, tham số mốc khởi tạo (θ = 0.15, lệch J = 0)
  v3-init     năng lượng chỉ V1–V3, tham số khởi tạo
  v2-learned  như v2 nhưng tham số mốc học được khi huấn luyện CBM-Sign trên cả 3 miền Brugada
  v3-learned  như v3, tham số học được
  v2-hard / v3-hard   ngưỡng cứng (lần đầu năng lượng < θ), để kiểm tra dạng mềm không gây lệch
Kèm theo: điểm "cất cánh" riêng của V2 (Jl, nơi đo J_elev), và tương quan QRS_w với độ rộng QRS
do bác sĩ đánh dấu.

Khác biệt nhỏ so với harmonize.py: LUDB có hệ số khuếch đại khác nhau giữa các chuyển đạo, nên
mỗi chuyển đạo được chia cho MAD trước khi dò đỉnh/căn chỉnh. Các khái niệm bất biến affine theo
từng chuyển đạo nên phép chia này không ảnh hưởng tới giá trị khái niệm, chỉ ảnh hưởng nhẹ tới
việc dò đỉnh và căn chỉnh.

Chạy: python ludb_validate.py <thư mục LUDB (có ludb.csv, data/)> <harmonized> <output>
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, filtfilt, find_peaks, resample_poly
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cbm_lib4 as L  # noqa: E402
from concepts3 import bounded, interp  # noqa: E402

LD, HD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FS0, FS = 500, 100
PRE, POST, ANCHOR = 29, 46, 28
IDX = [0, 1, 2, 6, 7, 8, 9, 10, 11]            # i, ii, iii, v1..v6 (thứ tự đã hài hòa)
MATCH = 50                                      # ±100 ms ở 500 Hz
torch.set_num_threads(2)


# ------------------------------------------------------------------ tiền xử lý giống harmonize.py
def qrs_energy(beat):
    return int(np.argmax(np.abs(np.diff(beat, axis=1)).sum(0)))


def align(beat):
    shift = ANCHOR - qrs_energy(beat)
    if shift == 0:
        return beat, 0
    out = np.empty_like(beat)
    if shift > 0:
        out[:, shift:] = beat[:, :-shift]; out[:, :shift] = beat[:, :1]
    else:
        out[:, :shift] = beat[:, -shift:]; out[:, shift:] = beat[:, -1:]
    return out, shift


def zscore(b):
    return (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-8)


BA = butter(2, [0.5 / (FS0 / 2), 40 / (FS0 / 2)], btype="band")


def prep(rec):
    raw = np.fromfile(os.path.join(LD, "data", f"{rec}.dat"), dtype="<i2").reshape(-1, 12)[:, IDX].astype(float).T
    raw = raw - np.median(raw, 1, keepdims=True)
    raw = raw / (np.median(np.abs(raw), 1, keepdims=True) * 1.4826 + 1e-6)
    sig = resample_poly(filtfilt(*BA, raw, axis=1), 1, 5, axis=1)            # (9, 1000) ở 100 Hz
    env = np.abs(np.diff(sig, axis=1)).sum(0)
    peaks, _ = find_peaks(env, distance=int(0.33 * FS), height=0.35 * np.percentile(env, 99))
    peaks = np.array([p for p in peaks if p - PRE >= 0 and p + POST <= sig.shape[1]])
    med = np.median(np.stack([sig[:, p - PRE:p + POST] for p in peaks]), 0)
    a, shift = align(med)
    return zscore(a).astype(np.float32), peaks, shift


def annotations(rec, ext="v2"):
    a = wfdb.rdann(os.path.join(LD, "data", str(rec)), ext)
    s, y = a.sample, a.symbol
    out = []
    for i in range(1, len(y) - 1):
        if y[i] == "N" and y[i - 1] == "(" and y[i + 1] == ")":
            out.append((s[i - 1], s[i], s[i + 1]))
    return np.array(out)                       # (n, 3): onset, đỉnh, offset (mẫu 500 Hz)


# ------------------------------------------------------------------ dò điểm J
def soft_J(cm, x):
    with torch.no_grad():
        _, aux = cm(torch.tensor(x), return_aux=True)
    return aux["J"].numpy(), aux["qrs_w"].numpy()


def hard_J(cm, x, leads):
    """Lần đầu năng lượng chuẩn hóa < θ (không làm mềm, không lệch J)."""
    xv = x[:, leads]
    d = np.diff(xv, axis=2) / (xv.std(-1, keepdims=True) + 1e-6)
    e = np.sqrt(d * d + 1e-6).sum(1)
    e = e / e[:, ANCHOR - 6:ANCHOR + 7].max(1, keepdims=True)
    th = float(bounded(cm.p_theta, 0.03, 0.5).detach())
    J = []
    for r in e:
        idx = np.where(r[ANCHOR + 1:ANCHOR + 15] < th)[0]
        J.append(ANCHOR + 1 + (idx[0] if len(idx) else 13) + 1.0)
    return np.array(J)


def takeoff_V2(cm, x, J):
    """Điểm cất cánh riêng của V2 (cùng công thức trong forward)."""
    with torch.no_grad():
        xt = torch.tensor(x); b = cm.base(xt)[:, 4:5]; xl = xt[:, 4, :]
        tau = 0.02 * (xl.std(-1, keepdim=True) + 1e-6)
        P = torch.tensor(J)[:, None] + cm.k_to[None]
        yto = interp(xl, P) - b
        w = torch.softmax(yto / tau, 1)
        return (w * P).sum(1).numpy()


# ------------------------------------------------------------------ tham số mốc học được
def learned(cv):
    X, Y = L.load(HD)
    xtr, ytr, dtr, di = L.pack(X, Y, L.DOMAINS)
    tr, va = L.strat_val(ytr, di, 0)
    signs = L.sign_gate(xtr[tr], ytr[tr], dtr[tr], cv=cv)
    ms = [L.train_cbm2(xtr[tr], ytr[tr], dtr[tr], xtr[va], ytr[va], dtr[va], s, False, signs, cv=cv) for s in (0, 1)]
    return [m.concepts.eval() for m in ms]


meta = pd.read_csv(os.path.join(LD, "ludb.csv"))
txt = lambda c: meta[c].fillna("").astype(str).str.lower()
meta["paced"] = txt("Cardiac pacing").str.contains("pacing|pacemaker")
meta["rbbb"] = txt("Conduction abnormalities").str.contains("right bundle")
meta["crbbb"] = txt("Conduction abnormalities").str.contains("complete right bundle") & ~txt("Conduction abnormalities").str.contains("incomplete right bundle")
meta["lbbb"] = txt("Conduction abnormalities").str.contains("left bundle")
meta["af"] = txt("Rhythms").str.contains("fibrillation|flutter")
meta["conduction"] = txt("Conduction abnormalities").str.strip() != ""
meta = meta.set_index("ID")

recs = sorted(int(f[:-4]) for f in os.listdir(os.path.join(LD, "data")) if f.endswith(".dat")
              and os.path.exists(os.path.join(LD, "data", f[:-4] + ".v2")))
XB, PK, SH, AN, ok = [], [], [], [], []
for r in recs:
    try:
        x, pk, sh = prep(r); an = annotations(r)
    except Exception as e:                                  # noqa: BLE001
        print("bỏ qua", r, e); continue
    XB.append(x); PK.append(pk); SH.append(sh); AN.append(an); ok.append(r)
XB = np.stack(XB)
print(f"{len(ok)} bản ghi đọc được; {int(meta.loc[ok, 'paced'].sum())} có máy tạo nhịp")

variants = {}
for cv in ["v2", "v3"]:
    init = L.CLS[cv](learn=False).eval()
    leads = list(range(9)) if cv == "v2" else [3, 4, 5]
    J, qw = soft_J(init, XB)
    variants[f"{cv}-init"] = (J, qw, takeoff_V2(init, XB, J))
    variants[f"{cv}-hard"] = (hard_J(init, XB, leads), None, None)
    if os.environ.get("QUICK"):
        continue
    cms = learned(cv)
    outs = [soft_J(cm, XB) for cm in cms]
    Jm = np.mean([o[0] for o in outs], 0)
    variants[f"{cv}-learned"] = (Jm, np.mean([o[1] for o in outs], 0), np.mean([takeoff_V2(cm, XB, Jm) for cm in cms], 0))
    variants[f"{cv}-learned"] += ([cm.fiducial_values() for cm in cms],)


def record_errors(J):
    """Sai số (ms) của từng bản ghi + mốc tham chiếu so với đỉnh R chú thích."""
    err, ref, pred, qrsd, nb = [], [], [], [], []
    for i in range(len(ok)):
        an = AN[i]; e, rf, pr = [], [], []
        for p in PK[i]:
            if len(an) == 0:
                break
            t500 = p * 5
            j = np.argmin(np.abs(an[:, 1] - t500))
            if abs(an[j, 1] - t500) > MATCH:
                continue
            tJ = (p - PRE + J[i] - SH[i]) * 10.0
            e.append(tJ - an[j, 2] * 2.0); rf.append((an[j, 2] - an[j, 1]) * 2.0); pr.append(tJ - an[j, 1] * 2.0)
        err.append(np.median(e) if e else np.nan); ref.append(np.median(rf) if rf else np.nan)
        pred.append(np.median(pr) if pr else np.nan); nb.append(len(e))
        qrsd.append(np.median((an[:, 2] - an[:, 0]) * 2.0) if len(an) else np.nan)
    return np.array(err), np.array(ref), np.array(pred), np.array(qrsd), np.array(nb)


GROUPS = {"tất cả (không tạo nhịp)": lambda m: ~m.paced,
          "dẫn truyền bình thường": lambda m: ~m.paced & ~m.conduction,
          "có rối loạn dẫn truyền": lambda m: ~m.paced & m.conduction,
          "RBBB (hoàn toàn/không)": lambda m: ~m.paced & m.rbbb,
          "LBBB (hoàn toàn/không)": lambda m: ~m.paced & m.lbbb,
          "rung/cuồng nhĩ": lambda m: ~m.paced & m.af,
          "có máy tạo nhịp": lambda m: m.paced}


def stats(e):
    e = e[~np.isnan(e)]
    return dict(n=len(e), bias=round(e.mean(), 1), sd=round(e.std(ddof=1), 1) if len(e) > 1 else np.nan,
                mae=round(np.abs(e).mean(), 1), med_abs=round(np.median(np.abs(e)), 1),
                within10=round((np.abs(e) <= 10).mean() * 100, 1), within20=round((np.abs(e) <= 20).mean() * 100, 1))


rows, per_rec = [], pd.DataFrame(index=ok)
m = meta.loc[ok]
for name, v in variants.items():
    err, ref, pred, qrsd, nb = record_errors(v[0])
    per_rec[f"err_{name}"] = err
    per_rec["ref_off_minus_R"] = ref; per_rec["qrs_dur_ann"] = qrsd; per_rec["n_beats_matched"] = nb
    per_rec[f"pred_{name}"] = pred
    for g, f in GROUPS.items():
        r = dict(variant=name, group=g, **stats(err[f(m).values]))
        if v[1] is not None and g == "tất cả (không tạo nhịp)":
            sel = f(m).values & ~np.isnan(qrsd)
            r["spearman_QRSw_vs_QRSdur"] = round(spearmanr(v[1][sel], qrsd[sel])[0], 3)
        rows.append(r)
    if v[2] is not None:                                   # điểm cất cánh V2
        e2, *_ = record_errors(v[2])
        per_rec[f"err_takeoffV2_{name}"] = e2
        for g in ["tất cả (không tạo nhịp)", "dẫn truyền bình thường"]:
            rows.append(dict(variant=f"{name} · cất cánh V2", group=g, **stats(e2[GROUPS[g](m).values])))
for c in ["paced", "conduction", "rbbb", "lbbb", "af"]:
    per_rec[c] = m[c].values
res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(res.to_string(index=False))
res.to_csv(os.path.join(OUT, "ludb_J_summary.csv"), index=False)
per_rec.index.name = "record"; per_rec.round(2).to_csv(os.path.join(OUT, "ludb_J_per_record.csv"))
fid = {k: v[3] for k, v in variants.items() if len(v) > 3}
json.dump(dict(n_records=len(ok), n_paced=int(m.paced.sum()), learned_fiducials=fid), open(os.path.join(OUT, "ludb_J_meta.json"), "w"), indent=1)
print(json.dumps(fid, indent=1))

# ------------------------------------------------------------------ hình
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#9a9993", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb"})
np_ = ~m.paced.values
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for name, c, lb in [("v2-init", "#eb6834", "v2: năng lượng 9 chuyển đạo"), ("v3-init", "#2a78d6", "v3: năng lượng V1–V3"),
                    ("v3-learned", "#1baf7a", "v3, tham số học được")]:
    ref, pred = per_rec["ref_off_minus_R"].values[np_], per_rec[f"pred_{name}"].values[np_]
    ax[0].plot(ref, pred, "o", color=c, ms=4, alpha=0.6, label=lb)
    e = per_rec[f"err_{name}"].values[np_]
    ax[1].hist(e[~np.isnan(e)], bins=np.arange(-80, 82, 5), histtype="step", lw=2, color=c, label=lb)
lim = [20, 140]
ax[0].plot(lim, lim, color="#9a9993", lw=1); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("Bác sĩ: kết thúc QRS − đỉnh R (ms)"); ax[0].set_ylabel("Mô hình: điểm J − đỉnh R (ms)")
ax[0].set_title("Điểm J so với chú thích LUDB (mỗi điểm: 1 bản ghi)", loc="left", fontsize=11, color="#0b0b0b")
ax[1].axvspan(-11.6, 11.6, color="#e4e3df", zorder=0)
ax[1].set_xlabel("Sai số điểm J (ms), vùng xám: ±11.6 ms (dung sai CSE)"); ax[1].set_ylabel("Số bản ghi")
ax[1].set_title("Phân bố sai số", loc="left", fontsize=11, color="#0b0b0b")
for a in ax:
    a.grid(color="#e4e3df"); a.set_axisbelow(True)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3, fontsize=9)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(os.path.join(OUT, "fig_ludb_J.png"), dpi=150); plt.close()
print("xong")
