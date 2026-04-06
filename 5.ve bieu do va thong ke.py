import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# 1. Đọc dữ liệu
df = pd.read_csv('Final_Research_Data.csv')
df = df.dropna(subset=['V1_Slope'])

# 2. Vẽ biểu đồ so sánh độ dốc ST giữa người Sống và người Chết
plt.figure(figsize=(10, 6))
sns.boxplot(x='sudden_death', y='V1_Slope', data=df, palette='Set1')
plt.title('SO SÁNH ĐỘ DỐC ST (V1_SLOPE) GIỮA NHÓM SỐNG (0) VÀ ĐỘT TỬ (1)')
plt.ylabel('Độ dốc ST (mV/s)')
plt.xlabel('Trạng thái: 0 = Sống | 1 = Đột tử')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Final_Comparison.png')
plt.show()

# 3. Tính chỉ số P-value (Bằng chứng thống kê)
group_live = df[df['sudden_death'] == 0]['V1_Slope']
group_death = df[df['sudden_death'] == 1]['V1_Slope']
stat, p = mannwhitneyu(group_live, group_death)

print(f"--- KẾT QUẢ THỐNG KÊ ---")
print(f"Giá trị P-value: {p:.4f}")
if p < 0.05:
    print("✅ CỰC KỲ Ý NGHĨA: Độ dốc ST thực sự khác biệt giữa hai nhóm!")
else:
    print("⚠️ Chưa thấy sự khác biệt rõ rệt, cần phân tích sâu hơn nhóm Brugada ẩn.")