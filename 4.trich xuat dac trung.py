"""
BƯỚC 4: TRÍCH XUẤT ĐẶC TRƯNG CHO TOÀN BỘ 363 CA
Đầu ra: Final_Research_Data.csv = metadata + 14 đặc trưng ECG + 2 cột chất lượng + cột error.
Danh sách và ý nghĩa đặc trưng: ecg_pipeline.FEATURE_DESCRIPTIONS.
"""
import pandas as pd

import ecg_pipeline as ep

OUTPUT_PATH = 'Final_Research_Data.csv'

if __name__ == '__main__':
    df = pd.read_csv(ep.METADATA_PATH)
    print(f'🚀 Đang xử lý {len(df)} ca. Vui lòng đợi...')
    rows = []
    for idx, row in df.iterrows():
        feats, _ = ep.process_patient(row['patient_id'])
        rows.append({**row.to_dict(), **feats})
        if idx % 50 == 0:
            print(f'✅ Đã xong {idx}/{len(df)}')

    df_final = pd.DataFrame(rows)
    num_cols = ep.FEATURE_COLUMNS + ep.QUALITY_COLUMNS
    df_final[num_cols] = df_final[num_cols].round(4)
    df_final.to_csv(OUTPUT_PATH, index=False)

    failed = df_final[df_final['error'].fillna('') != '']
    print(f'\n🎉 Đã lưu {OUTPUT_PATH}: {len(df_final) - len(failed)} ca thành công, {len(failed)} ca lỗi.')
    for _, r in failed.iterrows():
        print(f"   ⚠️ ID {r['patient_id']}: {r['error']}")
    print('👉 Chạy bước 3 với các ID nghi ngờ để kiểm tra bằng mắt trước khi làm thống kê.')
