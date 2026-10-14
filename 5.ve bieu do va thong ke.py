import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

FEATURES = ['V1_J', 'V1_Slope', 'V1_T_Amp', 'V1_QRS']

# 1. Đọc dữ liệu
df = pd.read_csv('Final_Research_Data.csv')
df = df.dropna(subset=['V1_Slope'])

# 2. Vẽ biểu đồ so sánh độ dốc ST giữa người Sống và người Chết
plt.figure(figsize=(10, 6))
sns.boxplot(x='sudden_death', y='V1_Slope', hue='sudden_death', data=df,
            palette='Set1', legend=False)
plt.title('SO SÁNH ĐỘ DỐC ST (V1_SLOPE) GIỮA NHÓM SỐNG (0) VÀ ĐỘT TỬ (1)')
plt.ylabel('Độ dốc ST (mV/s)')
plt.xlabel('Trạng thái: 0 = Sống | 1 = Đột tử')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Final_Comparison.png')
plt.show()

# 3. Kiểm định Mann-Whitney cho từng đặc trưng
n_live = (df['sudden_death'] == 0).sum()
n_death = (df['sudden_death'] == 1).sum()
print(f"--- KẾT QUẢ THỐNG KÊ (Sống n={n_live}, Đột tử n={n_death}) ---")
# Hiệu chỉnh Bonferroni vì kiểm định nhiều đặc trưng cùng lúc
alpha = 0.05 / len(FEATURES)
for col in FEATURES:
    live = df.loc[df['sudden_death'] == 0, col].dropna()
    death = df.loc[df['sudden_death'] == 1, col].dropna()
    stat, p = mannwhitneyu(live, death)
    mark = "✅ có ý nghĩa" if p < alpha else "—"
    print(f"{col:10s} trung vị {live.median():8.3f} vs {death.median():8.3f}   p = {p:.4f}  {mark}")
print(f"(Ngưỡng sau hiệu chỉnh Bonferroni: p < {alpha:.4f})")
