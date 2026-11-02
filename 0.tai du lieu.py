"""
BƯỚC 0: TẢI BỘ DỮ LIỆU BRUGADA-HUCA (chỉ cần chạy 1 lần, ~15 MB)
Nguồn chính thức: https://physionet.org/content/brugada-huca/1.0.0/
Script này tải từ bản sao GitHub của tác giả (NahuelCostaCortez/brugada-huca), hữu ích khi
PhysioNet bị chặn (ví dụ trong phiên Claude chạy trên cloud). Dữ liệu giống hệt: đã đối chiếu
V1_J của 40 ca đầu với kết quả chạy trên bản PhysioNet, sai khác < 0.0001 mV.

Cách chạy:
    python "0.tai du lieu.py" [thư_mục_đích]
Sau đó đặt biến môi trường BRUGADA_DIR=thư_mục_đích trước khi chạy các bước 1-5.
"""
import csv
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MIRROR = 'https://raw.githubusercontent.com/NahuelCostaCortez/brugada-huca/main'


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, path)
            return
        except OSError:
            if attempt == 2:
                raise


if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'brugada-huca'
    meta_path = os.path.join(out_dir, 'metadata.csv')
    download(f'{MIRROR}/metadata.csv', meta_path)
    with open(meta_path, newline='') as f:
        ids = [row['patient_id'] for row in csv.DictReader(f)]

    jobs = [(f'{MIRROR}/files/{pid}/{pid}.{ext}', os.path.join(out_dir, 'files', pid, f'{pid}.{ext}'))
            for pid in ids for ext in ('hea', 'dat')]
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda job: download(*job), jobs))
    print(f'✅ Đã tải {len(ids)} bản ghi vào {os.path.abspath(out_dir)}')
    print(f'👉 Đặt BRUGADA_DIR={os.path.abspath(out_dir)} rồi chạy các bước 1-5.')
