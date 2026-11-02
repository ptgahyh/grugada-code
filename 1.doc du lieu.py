"""
BƯỚC 1: KIỂM TOÁN DỮ LIỆU (DATA AUDIT)
Đếm số ca theo nhãn, liệt kê ca đột tử, và kiểm tra mọi bản ghi có cùng fs / độ dài / thứ tự chuyển đạo.
"""
import os

import pandas as pd
import wfdb

import ecg_pipeline as ep

if __name__ == '__main__':
    # 1. Đọc Metadata
    df = pd.read_csv(ep.METADATA_PATH)

    # 2. Thống kê quân số
    print('--- [BƯỚC 1: DATA AUDIT REPORT] ---')
    print(pd.crosstab(df['brugada'], df['sudden_death'], margins=True))
    print('\nLưu ý: tài liệu PhysioNet mô tả brugada là nhãn nhị phân 0/1, nhưng metadata có giá trị 2 '
          f"({(df['brugada'] == 2).sum()} ca). Dự án coi brugada >= 1 là Brugada (khớp con số 76 ca của PhysioNet).")

    # 3. Danh sách ca đột tử (SCD+) để kiểm chứng chéo
    scd = df[df['sudden_death'] == 1]
    print(f'\nDanh sách {len(scd)} ID Đột tử (SCD+): {scd["patient_id"].tolist()}')
    scd.to_csv('Audit_SCD_List.csv', index=False)

    # 4. Kiểm tra định dạng tín hiệu
    fs_set, len_set, missing = set(), set(), []
    for pid in df['patient_id']:
        try:
            h = wfdb.rdheader(os.path.join(ep.FILES_DIR, str(pid), str(pid)))
        except FileNotFoundError:
            missing.append(pid)
            continue
        fs_set.add(h.fs)
        len_set.add(h.sig_len)
        if list(h.sig_name) != ep.LEADS:
            print(f'⚠️ ID {pid}: thứ tự chuyển đạo khác chuẩn: {h.sig_name}')
    print(f'\nTần số lấy mẫu: {fs_set} Hz | Số mẫu mỗi bản ghi: {len_set} | Thiếu file: {missing or "không"}')
