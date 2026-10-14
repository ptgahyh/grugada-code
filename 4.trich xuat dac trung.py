import pandas as pd

from brugada_utils import METADATA_PATH, extract_features

df = pd.read_csv(METADATA_PATH)
total = len(df)
print(f"🚀 Đang xử lý {total} ca. Vui lòng đợi...")

features_list = []
for idx, row in enumerate(df.itertuples(index=False), start=1):
    f = extract_features(row.patient_id)
    features_list.append({**row._asdict(), **f})
    if idx % 50 == 0 or idx == total:
        print(f"✅ Đã xong {idx}/{total}")

df_final = pd.DataFrame(features_list)
df_final.to_csv('Final_Research_Data.csv', index=False)

n_fail = df_final['V1_J'].isna().sum()
print(f"\n🎉 Hoàn thành. Số ca lỗi/không trích được: {n_fail}")
print("File 'Final_Research_Data.csv' đã sẵn sàng để làm Thống kê.")
