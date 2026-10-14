import pandas as pd

from brugada_utils import METADATA_PATH

# 1. Đọc Metadata
df = pd.read_csv(METADATA_PATH)

# 2. Thống kê quân số
print("--- [BƯỚC 1: DATA AUDIT REPORT] ---")
ctab = pd.crosstab(df['brugada'], df['sudden_death'], margins=True)
print(ctab)

# 3. Trích xuất danh sách ca Đột tử (SCD+) để kiểm chứng chéo
scd_ids = df[df['sudden_death'] == 1]['patient_id'].tolist()
print(f"\nDanh sách {len(scd_ids)} ID Đột tử (SCD+): {scd_ids}")

# Lưu file kiểm toán
df[df['sudden_death'] == 1].to_csv('Audit_SCD_List.csv', index=False)
