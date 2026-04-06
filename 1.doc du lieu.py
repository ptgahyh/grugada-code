import pandas as pd
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = r'D:\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0\brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0'
metadata_path = os.path.join(BASE_DIR, 'metadata.csv')

# 1. Đọc Metadata
df = pd.read_csv(metadata_path)

# 2. Thống kê quân số
print("--- [BƯỚC 1: DATA AUDIT REPORT] ---")
ctab = pd.crosstab(df['brugada'], df['sudden_death'], margins=True)
print(ctab)

# 3. Trích xuất danh sách 11 ca Đột tử (SCD+) để kiểm chứng chéo
scd_ids = df[df['sudden_death'] == 1]['patient_id'].tolist()
print(f"\nDanh sách 11 ID Đột tử (SCD+): {scd_ids}")

# Lưu file kiểm toán
df[df['sudden_death'] == 1].to_csv('Audit_SCD_List.csv', index=False)