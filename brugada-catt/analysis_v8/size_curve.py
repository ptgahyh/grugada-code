"""(Bước 8) Chỉ chạy phần 3 của v8_analyze.py (cỡ lô đích cần thiết), dùng chung các hàm.
Chạy: NSIZE=100 python size_curve.py <scores> <out>"""
import os, sys
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v8_analyze.py")).read()
head = src.split("# ---------------------------------------------------------------- 0.")[0]
tail = src.split("# ---------------------------------------------------------------- 3. cỡ lô đích cần thiết")[1]
exec(compile(head + "\nrng = np.random.default_rng(2027)\nHELD = [2, 3, 4, 5, 6, 7, 8]\n" + tail, "size_curve", "exec"))
