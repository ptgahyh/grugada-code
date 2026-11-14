"""Bước 27: như harmonized/harmonize.py, nhưng nhịp Melo được lọc thông thấp Butterworth bậc 2, 40 Hz, hai chiều ở 200 Hz
trước khi hạ mẫu (cùng thành phần thông thấp mà HUCA đã qua). HUCA chép nguyên từ harmonized/huca.npz. Xem PLAN_vi.md.

Chạy: python analysis_v27/harmonize_v27.py <harmonized> <thư mục melo_processed> <output>
"""
import os
import shutil
import sys

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

HD, MELO_DIR, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harmonized", "harmonize.py"), encoding="utf8").read()
exec("\n".join(l for l in SRC.split("\n") if l.startswith(("FS =", "PRE,", "ANCHOR =", "LEADS ="))))
for fn in ("def qrs_energy", "def align", "def zscore"):
    s = SRC.index(fn); exec(SRC[s:SRC.index("\n\n\n", s)])
shutil.copy(os.path.join(HD, "huca.npz"), os.path.join(OUT, "huca.npz"))
b, a = butter(2, 40 / (200 / 2), btype="low")
for name in ["claris", "mortara"]:
    d = np.load(os.path.join(MELO_DIR, f"melo_{name}.npz"))
    X = d["X"][:, :9].astype(float)
    X = filtfilt(b, a, X, axis=2)                # BƯỚC 27: thông thấp 40 Hz ở 200 Hz
    X = resample_poly(X, 1, 2, axis=2)
    X = np.stack([zscore(align(x)) for x in X]).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), X=X, y=d["y"].astype(np.int8), ids=np.arange(len(X)), leads=np.array(LEADS), fs=FS)
    print(name, X.shape, "Brugada =", int(d["y"].sum()))
