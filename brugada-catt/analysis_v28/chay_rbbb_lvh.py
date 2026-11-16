"""Chạy toàn bộ các bước RBBB/LVH (14, 15, 22, 28) của dự án Brugada/CATT ngay trên máy này.

Cách chạy (PowerShell hoặc cmd, trong thư mục chứa tệp này):
    python chay_rbbb_lvh.py

Cần trong cùng thư mục:
    brugada_code_15_11.zip                    (code, Claude đã đặt sẵn)
    physionet-ecg-arrhythmia-1.0.0.zip        (Chapman/Ningbo, đọc thẳng từ zip, không cần giải nén)
    archive (2).zip.part1 ... part6           (PTB-XL đã cắt; script tự ghép và chỉ giải nén phần cần)

Chạy lại được: bước nào xong rồi thì bỏ qua. Bị ngắt (tắt máy, mất điện) cứ chạy lại đúng lệnh trên.
Kết quả tổng hợp (không có dữ liệu từng bệnh nhân) nằm ở thư mục ket_qua_15_11. Nhật ký: ket_qua_15_11/log.txt
"""
import glob
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, "_chay_15_11")
CODE = os.path.join(WORK, "code")
RES = os.path.join(BASE, "ket_qua_15_11")
PTB = os.path.join(WORK, "ptbxl")
CN_ZIP = os.path.join(BASE, "physionet-ecg-arrhythmia-1.0.0.zip")
NPZ14, NPZ15 = os.path.join(WORK, "npz_rbbb"), os.path.join(WORK, "npz_lvh")
SC14, SC15, SC22 = (os.path.join(WORK, d) for d in ("scores_rbbb", "scores_lvh", "scores_lvh_mv"))
LOG = os.path.join(RES, "log.txt")
def _ram_gb():
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 2 ** 30
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
        except Exception:
            return 8.0


RAM_GB = _ram_gb()
# mỗi tiến trình huấn luyện dùng 1 luồng CPU và tối đa khoảng 1,5 GB RAM
NPROC = int(os.environ.get("NPROC", max(2, min(8, (os.cpu_count() or 4) // 2, int(RAM_GB // 2)))))
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", NPROC=str(NPROC), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
PY = sys.executable
for _s in (sys.stdout, sys.stderr):          # tránh lỗi mã hóa khi in tiếng Việt ra cửa sổ lệnh Windows
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
T0 = time.time()
WIN = os.name == "nt"
CHILDREN = []          # mọi tiến trình con đang chạy, để dừng sạch khi cần
_LAST_INT = [0.0]


def keep_awake(on=True):
    """Không cho Windows tự ngủ khi đang chạy (gập máy hay tắt máy thì vẫn dừng)."""
    if WIN:
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
        except Exception:
            pass


def start(args, logf):
    """Tiến trình con ở nhóm riêng: Ctrl+C trong cửa sổ lệnh không giết nó (chỉ chương trình chính quyết định dừng)."""
    kw = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if WIN else dict(start_new_session=True)
    p = subprocess.Popen([PY] + args, cwd=CODE, env=ENV, stdout=logf, stderr=subprocess.STDOUT, **kw)
    CHILDREN.append(p)
    return p


def kill_children():
    for p in CHILDREN:
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass


def on_ctrl_c(signum, frame):
    now = time.time()
    if now - _LAST_INT[0] < 15:
        say("Nhận Ctrl+C lần hai: dừng theo yêu cầu. Chạy lại đúng lệnh cũ để làm tiếp từ chỗ dừng.")
        kill_children(); keep_awake(False); os._exit(2)
    _LAST_INT[0] = now
    say("Nhận Ctrl+C (có thể do bấm nhầm, ví dụ khi sao chép chữ). VẪN TIẾP TỤC CHẠY. Muốn dừng thật: bấm Ctrl+C lần nữa trong 15 giây.")


def wait_all(procs):
    """Chờ các tiến trình con bằng cách hỏi định kỳ (để Ctrl+C được xử lý ngay)."""
    while any(p.poll() is None for p in procs):
        time.sleep(2)


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} ph] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fail(msg, logfile=None):
    say("LỖI: " + msg)
    if logfile and os.path.exists(logfile):
        with open(logfile, encoding="utf-8", errors="replace") as f:
            tail = f.read()[-3000:]
        print("----- cuối nhật ký bước lỗi -----\n" + tail, flush=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(tail + "\n")
    say("Dừng. Hãy nhắn Claude \"bị lỗi\" (Claude sẽ đọc ket_qua_15_11/log.txt). Sửa xong chạy lại đúng lệnh cũ, các bước đã xong sẽ được bỏ qua.")
    sys.exit(1)


def run(name, args_list, done):
    """Chạy một nhóm lệnh song song (mỗi phần tử là danh sách đối số sau `python`). Bỏ qua nếu tệp `done` đã có."""
    if os.path.exists(done):
        say(f"bỏ qua (đã xong): {name}"); return
    say(f"bắt đầu: {name}")
    procs = []
    for k, a in enumerate(args_list):
        lf = os.path.join(RES, "logs", f"{name}_{k}.log".replace(" ", "_"))
        f = open(lf, "w", encoding="utf-8")
        procs.append((start(a, f), f, lf))
    wait_all([p for p, _, _ in procs])
    for p, f, lf in procs:
        rc = p.wait(); f.close()
        if rc != 0:
            for q, _, _ in procs:
                if q.poll() is None:
                    q.kill()
            fail(f"bước '{name}' thoát với mã {rc}", lf)
    if not os.path.exists(done):
        fail(f"bước '{name}' chạy xong nhưng không thấy tệp {done}", procs[0][2])
    say(f"xong: {name}")


class Parts:
    """Đọc nhiều phần cắt nối tiếp như một tệp (để mở zip PTB-XL mà không phải ghép lại trên đĩa)."""

    def __init__(self, paths):
        self.paths = paths; self.sizes = [os.path.getsize(p) for p in paths]; self.total = sum(self.sizes); self.pos = 0
        self.files = [open(p, "rb") for p in paths]

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.total + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.total - self.pos
        out = []; pos = self.pos
        while n > 0 and pos < self.total:
            i, start = 0, 0
            while pos >= start + self.sizes[i]:
                start += self.sizes[i]; i += 1
            f = self.files[i]; f.seek(pos - start); b = f.read(min(n, self.sizes[i] - (pos - start)))
            if not b:
                break
            out.append(b); pos += len(b); n -= len(b)
        self.pos = pos
        return b"".join(out)

    def close(self):
        for f in self.files:
            f.close()


def prepare_ptb():
    done = os.path.join(PTB, "_XONG.txt")
    if os.path.exists(done):
        say("bỏ qua (đã xong): giải nén PTB-XL"); return
    full = os.path.join(BASE, "archive (2).zip")
    if os.path.exists(full):
        src = open(full, "rb")
    else:
        parts = [os.path.join(BASE, f"archive (2).zip.part{i}") for i in range(1, 50)]
        parts = [p for p in parts if os.path.exists(p)]
        if not parts:
            fail("không thấy 'archive (2).zip' hay các tệp 'archive (2).zip.part1..6' trong " + BASE)
        say(f"PTB-XL: ghép {len(parts)} phần, {sum(os.path.getsize(p) for p in parts) / 1e9:.2f} GB")
        src = Parts(parts)
    say("bắt đầu: giải nén PTB-XL (chỉ records100 và ptbxl_database.csv)")
    try:
        z = zipfile.ZipFile(src)
    except zipfile.BadZipFile as e:
        fail(f"không mở được zip PTB-XL ({e}). Có thể thiếu một phần cắt.")
    names = z.namelist()
    csv = [n for n in names if n.endswith("ptbxl_database.csv")]
    if not csv:
        fail("zip PTB-XL không có ptbxl_database.csv")
    pre = csv[0][:-len("ptbxl_database.csv")]
    want = [n for n in names if n == csv[0] or (n.startswith(pre + "records100/") and not n.endswith("/"))]
    os.makedirs(PTB, exist_ok=True)
    for k, n in enumerate(want):
        out = os.path.join(PTB, *n[len(pre):].split("/"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(n) as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        if k % 10000 == 0:
            say(f"  PTB-XL {k}/{len(want)} tệp")
    z.close(); src.close()
    open(done, "w").write(str(len(want)))
    say(f"xong: giải nén PTB-XL ({len(want)} tệp)")


def chunks(script, *args):
    return [[script, *args, str(c), str(NPROC)] for c in range(NPROC)]


def n_scores(d):
    return len(glob.glob(os.path.join(d, "scores_*_*.npz")))


def train(name, script, npz, sc):
    marker = os.path.join(sc, "_XONG.txt")
    if not os.path.exists(marker):
        os.makedirs(sc, exist_ok=True)
        import numpy as np
        for fn in glob.glob(os.path.join(sc, "scores_*_*.npz")):      # tệp ghi dở khi bị ngắt: xóa để chạy lại lượt đó
            try:
                with np.load(fn) as z:
                    _ = [z[k].shape for k in z.files]
            except Exception:
                os.remove(fn); say(f"  xóa tệp điểm ghi dở: {os.path.basename(fn)}")
        say(f"bắt đầu: {name} ({n_scores(sc)}/18 lượt đã có, {NPROC} tiến trình)")
        procs = []
        for k, a in enumerate(chunks(script, npz, sc)):
            lf = os.path.join(RES, "logs", f"{name}_{k}.log".replace(" ", "_"))
            f = open(lf, "a", encoding="utf-8")
            procs.append((start(a, f), f, lf))
        last, t = -1, 0
        while any(p.poll() is None for p, _, _ in procs):
            time.sleep(2); t += 2
            if t % 60 == 0:
                k = n_scores(sc)
                if k != last:
                    say(f"  {name}: {k}/18 lượt"); last = k
        for p, f, lf in procs:
            f.close()
            if p.returncode != 0:
                fail(f"'{name}' thoát với mã {p.returncode}", lf)
        if n_scores(sc) != 18:
            fail(f"'{name}' chỉ có {n_scores(sc)}/18 tệp điểm", procs[0][2])
        open(marker, "w").write("18")
        say(f"xong: {name}")
    else:
        say(f"bỏ qua (đã xong): {name}")


def main():
    os.makedirs(os.path.join(RES, "logs"), exist_ok=True); os.makedirs(WORK, exist_ok=True)
    say(f"Python {sys.version.split()[0]}, {os.cpu_count()} nhân, RAM {RAM_GB:.0f} GB, dùng {NPROC} tiến trình. Thư mục: {BASE}")
    say("Đang giữ máy không tự ngủ. Đừng gập máy/tắt máy. Ctrl+C một lần sẽ bị bỏ qua; muốn dừng thì bấm hai lần.")
    free = shutil.disk_usage(BASE).free / 1e9
    say(f"Ổ đĩa còn trống {free:.1f} GB (cần khoảng 4 GB)")
    if free < 4:
        fail("ổ đĩa còn dưới 4 GB trống")
    # ---- 0. tệp đầu vào
    codezip = os.path.join(BASE, "brugada_code_15_11.zip")
    for p in (codezip, CN_ZIP):
        if not os.path.exists(p):
            fail("thiếu tệp " + p)
    # ---- 1. gói Python
    need = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "sklearn": "scikit-learn", "wfdb": "wfdb", "torch": "torch"}
    miss = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
    if miss:
        say("cài gói còn thiếu: " + " ".join(miss) + " (torch khoảng 200 MB, cần mạng)")
        rc = subprocess.call([PY, "-m", "pip", "install", *miss])
        if rc != 0:
            fail("pip install không thành công. Thử chạy tay: python -m pip install " + " ".join(miss))
    # ---- 2. code
    if os.path.exists(CODE):
        shutil.rmtree(CODE)
    with zipfile.ZipFile(codezip) as z:
        z.extractall(CODE)
    say("đã giải nén code")
    # ---- 2b. phiên bản dữ liệu: bản 2 (15/11 tối) bỏ bản ghi Chapman lặp; kết quả của bản trước phải bỏ
    ver = os.path.join(WORK, "_phien_ban_2.txt")
    if not os.path.exists(ver):
        for d in (NPZ14, NPZ15, SC14, SC15, SC22, *(os.path.join(RES, x) for x in ("v14_rbbb", "v15_lvh", "v22_lvh_mv", "v28"))):
            if os.path.exists(d):
                shutil.rmtree(d); say("xóa kết quả của bản script cũ: " + d)
        open(ver, "w").write("2")
    # ---- 3. PTB-XL
    prepare_ptb()
    # ---- 3b. bản 3 (16/11): khi một bản sao của bản ghi Chapman hỏng thì thử bản sao khác.
    #      Chuẩn hóa lại vào thư mục tạm; chỉ khi số bản ghi đổi mới bỏ kết quả huấn luyện cũ.
    ver3 = os.path.join(WORK, "_phien_ban_3.txt")
    if not os.path.exists(ver3):
        old14, old15 = os.path.join(NPZ14, "harmonize_log.json"), os.path.join(NPZ15, "harmonize_log.json")
        if os.path.exists(old14) and os.path.exists(old15):
            import json
            t14, t15 = NPZ14 + "_v3", NPZ15 + "_v3"
            run("chuan_hoa_rbbb_v3", [["analysis_v14/harmonize_rbbb.py", PTB, CN_ZIP, t14]], os.path.join(t14, "harmonize_log.json"))
            run("chuan_hoa_lvh_v3", [["analysis_v15/harmonize_lvh.py", PTB, CN_ZIP, t15]], os.path.join(t15, "harmonize_log.json"))
            same = True
            for old, new, lab in ((old14, t14, "RBBB"), (old15, t15, "LVH")):
                a, b = json.load(open(old))["cn"], json.load(open(os.path.join(new, "harmonize_log.json")))
                say(f"  {lab} Chapman/Ningbo: trước {a['n']} bản ghi ({a['n_pos']} dương tính), nay {b['cn']['n']} ({b['cn']['n_pos']}); "
                    f"cứu được từ bản sao khác {b.get('cn_read_other_copy')}, hỏng mọi bản {b.get('cn_unreadable_all_copies')}")
                same &= (a["n"], a["n_pos"]) == (b["cn"]["n"], b["cn"]["n_pos"])
            if same:
                shutil.copy(os.path.join(t14, "harmonize_log.json"), old14); shutil.copy(os.path.join(t15, "harmonize_log.json"), old15)
                shutil.rmtree(t14); shutil.rmtree(t15)
                say("  số bản ghi không đổi: giữ kết quả huấn luyện đã có")
            else:
                for d in (SC14, SC15, SC22, *(os.path.join(RES, x) for x in ("v14_rbbb", "v15_lvh", "v22_lvh_mv", "v28"))):
                    if os.path.exists(d):
                        shutil.rmtree(d); say("  có thêm bản ghi đọc được: xóa kết quả cũ " + d)
                shutil.rmtree(NPZ14); shutil.rmtree(NPZ15); os.rename(t14, NPZ14); os.rename(t15, NPZ15)
        open(ver3, "w").write("3")
    # ---- 4. chuẩn hóa (đọc thẳng zip Chapman)
    run("chuan_hoa_rbbb", [["analysis_v14/harmonize_rbbb.py", PTB, CN_ZIP, NPZ14]], os.path.join(NPZ14, "harmonize_log.json"))
    run("chuan_hoa_lvh", [["analysis_v15/harmonize_lvh.py", PTB, CN_ZIP, NPZ15]], os.path.join(NPZ15, "harmonize_log.json"))
    r14, r15, r22, r28 = (os.path.join(RES, d) for d in ("v14_rbbb", "v15_lvh", "v22_lvh_mv", "v28"))
    for d, src in ((r14, NPZ14), (r15, NPZ15)):
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(src, "harmonize_log*.json")):
            shutil.copy(f, d)
    # ---- 5. huấn luyện + chấm điểm (phần lâu nhất)
    train("huan_luyen_rbbb", "analysis_v14/run_rbbb.py", NPZ14, SC14)
    train("huan_luyen_lvh", "analysis_v15/run_lvh.py", NPZ15, SC15)
    train("huan_luyen_lvh_mV", "analysis_v22/run_lvh_mv.py", NPZ15, SC22)
    # ---- 6. phân tích bước 14, 15, 22 (script cũ, không đổi)
    cher = os.path.join(CODE, "analysis_v28", "chernoff_samples.npy")
    for st in ("p1", "p3", "p4"):
        out = {"p1": "p1_spread.csv", "p3": "p3_stopping.csv", "p4": "p4_point.csv"}[st]
        run(f"b14_{st}", [["analysis_v14/analyze_rbbb.py", SC14, cher, r14, st]], os.path.join(r14, out))
        run(f"b15_{st}", [["analysis_v15/analyze_lvh.py", SC15, r15, st]], os.path.join(r15, out))
        run(f"b22_{st}", [["analysis_v15/analyze_lvh.py", SC22, r22, st]], os.path.join(r22, out))
    run("b14_tong_hop", [["analysis_v14/analyze_rbbb.py", SC14, cher, r14, "summary"]], os.path.join(r14, "summary.json"))
    run("b15_tong_hop", [["analysis_v15/analyze_lvh.py", SC15, r15, "summary"]], os.path.join(r15, "summary.json"))
    run("b22_tong_hop", [["analysis_v15/analyze_lvh.py", SC22, r22, "summary"]], os.path.join(r22, "summary.json"))
    # ---- 7. bước 28 (ngưỡng làm trơn, cỡ mẫu, cổng)
    for prob, sc in (("rbbb", SC14), ("lvh", SC15), ("lvhmv", SC22)):
        for st, out in (("a", f"a_spread_{prob}.csv"), ("b", f"b_plans_{prob}.csv"), ("c", f"c_gate_{prob}.csv"),
                        ("summary", f"summary_v28_{prob}.json")):
            run(f"b28_{prob}_{st}", [["analysis_v28/run28.py", sc, r28, prob, st]], os.path.join(r28, out))
    open(os.path.join(RES, "XONG.txt"), "w", encoding="utf-8").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    say("TẤT CẢ ĐÃ XONG. Nhắn Claude \"đã chạy xong\" để Claude đọc thư mục ket_qua_15_11.")


if __name__ == "__main__":
    import atexit
    import signal
    atexit.register(kill_children)
    signal.signal(signal.SIGINT, on_ctrl_c)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, on_ctrl_c)
    keep_awake(True)
    try:
        main()
    except SystemExit:
        kill_children(); keep_awake(False); raise
    except BaseException as e:
        say(f"LỖI không lường trước: {type(e).__name__}: {e}")
        kill_children(); keep_awake(False); raise
    keep_awake(False)
