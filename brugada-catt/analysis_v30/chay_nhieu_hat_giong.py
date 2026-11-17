"""Bước 30: huấn luyện lại RBBB, LVH, LVH mV thêm 4 lần với hạt giống khác (SEED_OFFSET = 1000·r, r = 1..4).

Cách chạy (PowerShell, trong thư mục D:\\brugada-huca-...-1.0.0):
    python chay_nhieu_hat_giong.py

Cần: đã chạy xong chay_rbbb_lvh.py (dùng lại dữ liệu chuẩn hóa trong _chay_15_11), và trong cùng thư mục có
chay_rbbb_lvh.py, brugada_code_17_11.zip. Lần 0 là lần chạy 16/11 (không chạy lại).

Như chay_rbbb_lvh.py: chạy lại được (bước xong thì bỏ qua), giữ máy không tự ngủ, một lần Ctrl+C bị bỏ qua.
Kết quả tổng hợp: ket_qua_15_11\\v30\\rep<r>\\{rbbb,lvh,lvhmv}\\p4_point.csv, p4_boot.csv. Nhật ký: ket_qua_15_11\\log.txt
"""
import os
import shutil
import signal
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chay_rbbb_lvh as B  # noqa: E402  (dùng lại các hàm: say, fail, run, train, giữ máy thức, xử lý Ctrl+C)

import numpy as np  # noqa: E402

REPS = tuple(int(x) for x in os.environ.get("REPS30", "1,2,3,4").split(","))   # REPS30 chỉ để thử
CODEZIP = os.path.join(B.BASE, "brugada_code_17_11.zip")
R30 = os.path.join(B.WORK, "r30")
OUT30 = os.path.join(B.RES, "v30")
PROBS = (  # (tên, script huấn luyện, npz, thư mục điểm lần 0, thư mục P4 lần 0)
    ("rbbb", "analysis_v14/run_rbbb.py", B.NPZ14, B.SC14, os.path.join(B.RES, "v14_rbbb")),
    ("lvh", "analysis_v15/run_lvh.py", B.NPZ15, B.SC15, os.path.join(B.RES, "v15_lvh")),
    ("lvhmv", "analysis_v22/run_lvh_mv.py", B.NPZ15, B.SC22, os.path.join(B.RES, "v22_lvh_mv")),
)


def check_differs(sc_new, sc0, label):
    """Lần mới phải khác lần 0 (nếu không, SEED_OFFSET không có tác dụng) và cùng nhãn, cùng cách chia."""
    for fn in sorted(os.listdir(sc0)):
        if not (fn.startswith("scores_") and fn.endswith(".npz")):
            continue
        a, b = np.load(os.path.join(sc0, fn)), np.load(os.path.join(sc_new, fn))
        for k in ("y_tgt", "y_val"):
            if not np.array_equal(a[k], b[k]):
                B.fail(f"{label} {fn}: nhãn '{k}' khác lần 0 (dữ liệu hoặc cách chia bị đổi)")
        if np.array_equal(a["f_tgt"], b["f_tgt"]):
            B.fail(f"{label} {fn}: điểm số TRÙNG lần 0, tức hạt giống không đổi. Kiểm lại brugada_code_17_11.zip")
    B.say(f"  kiểm {label}: cùng nhãn và cách chia với lần 0, điểm số khác lần 0 (đúng)")


def main():
    os.makedirs(os.path.join(B.RES, "logs"), exist_ok=True); os.makedirs(OUT30, exist_ok=True)
    B.say(f"=== BƯỚC 30: huấn luyện lại {len(REPS)} lần. Python {sys.version.split()[0]}, RAM {B.RAM_GB:.0f} GB, {B.NPROC} tiến trình")
    B.say("Đang giữ máy không tự ngủ. Đừng gập máy/tắt máy. Ctrl+C một lần sẽ bị bỏ qua; muốn dừng thì bấm hai lần.")
    for need in (os.path.join(B.WORK, "_phien_ban_3.txt"), os.path.join(B.NPZ14, "cn.npz"), os.path.join(B.NPZ15, "cn.npz"),
                 *(os.path.join(sc0, "_XONG.txt") for _, _, _, sc0, _ in PROBS), CODEZIP):
        if not os.path.exists(need):
            B.fail("thiếu " + need + " (cần chạy xong chay_rbbb_lvh.py và có brugada_code_17_11.zip cùng thư mục)")
    if os.path.exists(B.CODE):
        shutil.rmtree(B.CODE)
    with zipfile.ZipFile(CODEZIP) as z:
        z.extractall(B.CODE)
    if "SEED_OFFSET" not in open(os.path.join(B.CODE, "concepts_v4", "cbm_lib4.py"), encoding="utf-8").read():
        B.fail("code trong brugada_code_17_11.zip chưa có SEED_OFFSET")
    B.say("đã giải nén code (có SEED_OFFSET)")
    cher = os.path.join(B.CODE, "analysis_v28", "chernoff_samples.npy")
    for name, _, _, _, p40 in PROBS:                     # lần 0: chép P4 đã có
        d = os.path.join(OUT30, "rep0", name); os.makedirs(d, exist_ok=True)
        for f in ("p4_point.csv", "p4_boot.csv"):
            shutil.copy(os.path.join(p40, f), d)
    for r in REPS:
        B.ENV["SEED_OFFSET"] = str(1000 * r)
        B.say(f"--- lần {r}/{REPS[-1]} (SEED_OFFSET = {1000 * r})")
        for name, script, npz, sc0, _ in PROBS:
            sc = os.path.join(R30, f"rep{r}", name)
            B.train(f"r{r}_huan_luyen_{name}", script, npz, sc)
            check_differs(sc, sc0, f"lần {r} {name}")
        for name, _, _, _, _ in PROBS:
            sc = os.path.join(R30, f"rep{r}", name); out = os.path.join(OUT30, f"rep{r}", name); os.makedirs(out, exist_ok=True)
            args = (["analysis_v14/analyze_rbbb.py", sc, cher, out, "p4"] if name == "rbbb"
                    else ["analysis_v15/analyze_lvh.py", sc, out, "p4"])
            B.run(f"r{r}_p4_{name}", [args], os.path.join(out, "p4_boot.csv"))
        B.say(f"--- xong lần {r}")
    open(os.path.join(OUT30, "XONG.txt"), "w", encoding="utf-8").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    B.say("BƯỚC 30 ĐÃ XONG. Nhắn Claude \"đã chạy xong\" để Claude đọc thư mục ket_qua_15_11\\v30.")


if __name__ == "__main__":
    import atexit
    atexit.register(B.kill_children)
    signal.signal(signal.SIGINT, B.on_ctrl_c)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, B.on_ctrl_c)
    B.keep_awake(True)
    try:
        main()
    except SystemExit:
        B.kill_children(); B.keep_awake(False); raise
    except BaseException as e:
        B.say(f"LỖI không lường trước: {type(e).__name__}: {e}")
        B.kill_children(); B.keep_awake(False); raise
    B.keep_awake(False)
