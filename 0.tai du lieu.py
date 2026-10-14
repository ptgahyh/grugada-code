"""Tải bộ dữ liệu Brugada-HUCA (PhysioNet) về máy.

Mặc định lưu vào thư mục data/ cạnh các script (hoặc BRUGADA_DIR nếu đặt biến môi trường).
Nguồn: bản sao chính thức của PhysioNet trên AWS Open Data.

    python "0.tai du lieu.py"
"""
import hashlib
import os
import urllib.request
import zipfile

from brugada_utils import BASE_DIR

BASE_URL = 'https://physionet-open.s3.amazonaws.com/brugada-huca/1.0.0/'
FILES = ['metadata.csv', 'metadata_dictionary.csv', 'RECORDS', 'files.zip']


def download(name, dest):
    with urllib.request.urlopen(BASE_URL + name, timeout=60) as resp, open(dest, 'wb') as out:
        out.write(resp.read())


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"📁 Thư mục dữ liệu: {BASE_DIR}")

    with urllib.request.urlopen(BASE_URL + 'SHA256SUMS.txt', timeout=60) as resp:
        sums = dict(line.split()[::-1] for line in resp.read().decode().splitlines() if line.strip())

    for name in FILES:
        dest = os.path.join(BASE_DIR, name)
        print(f"⬇️  {name} ...", end=' ', flush=True)
        download(name, dest)
        with open(dest, 'rb') as f:
            ok = hashlib.sha256(f.read()).hexdigest() == sums.get(name)
        print("OK" if ok else "⚠️ SHA256 không khớp")
        if not ok:
            raise SystemExit(f"File {name} bị lỗi khi tải, hãy chạy lại.")

    zip_path = os.path.join(BASE_DIR, 'files.zip')
    with zipfile.ZipFile(zip_path) as z:
        # bỏ file rác của macOS (__MACOSX/, .DS_Store)
        members = [m for m in z.namelist()
                   if m.startswith('files/') and not os.path.basename(m).startswith('.')]
        z.extractall(BASE_DIR, members=members)
    os.remove(zip_path)

    n = len([d for d in os.listdir(os.path.join(BASE_DIR, 'files')) if not d.startswith('.')])
    print(f"🎉 Đã tải và giải nén {n} bản ghi.")


if __name__ == '__main__':
    main()
