# Rà soát y văn và mức độ chặt của chứng minh cho hai công thức của bước 11 (03/11/2026)

**Hai công thức cần kiểm tính mới:**
- **(A) Luật n_eff:** CATT với n mẫu không nhãn tương đương ngưỡng Youden với n_eff = n·π(1−π)(1−α−β)²/[q(1−q)] mẫu có nhãn.
- **(B) Thiên lệch do lỗi dồn ở ranh giới:** khi lỗi nhãn giả không độc lập, ngưỡng CATT là nghiệm của Σₖ πₖ fₖ(t)[aₖ(t) − q] = 0; xấp xỉ bậc nhất: độ lệch ≈ Γ/(D′ − Γ′).

**Kết luận ngắn:** sau khi rà 8 nhánh y văn liên quan, **chưa tìm thấy công trình nào công bố đúng hai công thức này**. Nhưng **các viên gạch để xây chúng đều đã có**, nên bài phải trích đúng và mô tả đóng góp ở mức "mở rộng và kết hợp", không phải "phát minh từ đầu". Rà soát này là tìm kiếm có hệ thống trên web, không phải tổng quan hệ thống chính thức; vẫn có thể có công trình chưa tìm thấy.

---

## 1. Kết quả rà soát theo từng nhánh

### 1.1. Học với nhãn nhiễu phụ thuộc lớp (class-conditional noise)
| Công trình | Nội dung | Liên quan |
|---|---|---|
| Menon và cs. 2015, ICML | Bộ phân loại tối ưu balanced error (ngưỡng Youden) bất biến với nhiễu nhãn phụ thuộc lớp | Là nền của CATT; trường hợp riêng của công thức (B) khi độc lập |
| Scott, Blanchard, Handy 2013 (COLT); Blanchard và cs. 2016 (Electronic Journal of Statistics), "Classification with asymmetric label noise" | Mô hình trộn lẫn (mutual contamination); kiểm định tỉ số khả dĩ giữ nguyên ROC; balanced error là tiêu chí duy nhất không bị ảnh hưởng; tốc độ hội tụ của ước lượng tỉ lệ trộn ~√(log n / n); hệ số (1 − π₀ − π₁) nhân vào rủi ro thừa | Gần nhất về mặt lý thuyết. **Mục 8 của họ dùng chính lập luận co-training: nhãn từ góc nhìn thứ nhất là nhiễu phụ thuộc lớp cho góc nhìn thứ hai nếu hai góc nhìn độc lập có điều kiện.** Họ **không** phân tích khi vi phạm độc lập, không có luật cỡ mẫu cho ngưỡng Youden |
| Natarajan và cs. 2013, NeurIPS | Ước lượng không chệch của hàm mất mát dưới nhiễu nhãn; chặn sai số tỉ lệ với 1/(1 − ρ₊ − ρ₋) | Nguồn gốc của hệ số (1 − α − β) trong các chặn sai số |
| Angluin và Laird 1988 | Số mẫu cần cho học PAC với nhiễu đối xứng tỉ lệ 1/(1 − 2η)² | Cùng tinh thần với (1 − α − β)² trong n_eff; cần trích |

**Hàm ý cho (A):** hệ số (1 − α − β)² không mới về tinh thần (Angluin–Laird, Natarajan, Blanchard). Điều mới là: (i) hằng số chính xác gồm cả π(1−π)/[q(1−q)]; (ii) cho bài toán chọn **ngưỡng Youden** (tốc độ n^(−1/3), không phải n^(−1/2)); (iii) phát biểu dạng "tương đương phân phối" giữa CATT với n mẫu và Youden với n_eff mẫu.

### 1.2. Nhiễu nhãn phụ thuộc mẫu (instance-dependent) và nhiễu dồn ở ranh giới
| Công trình | Nội dung | Liên quan |
|---|---|---|
| Menon, van Rooyen, Natarajan 2018, Machine Learning, "Learning from binary labels with instance-dependent noise" | Định nghĩa **boundary-consistent noise** (nhiễu lớn hơn ở mẫu khó, gần ranh giới); chứng minh **xếp hạng (AUROC) được giữ**; ghi rõ **ngưỡng khác 1/2 (balanced error, F-score) không còn giữ được** và cần biết hàm nhiễu ρ(x) | **Gần nhất với (B).** Họ chỉ nói ngưỡng *không* giữ được, **không** cho công thức độ lệch. Nhiễu của họ gắn với ranh giới *thật*, không gắn với điểm của một mô hình khác |
| Oyen và cs. 2022, NeurIPS, "Robustness to label noise depends on the shape of the noise distribution in feature space" | Nhiễu dồn ở ranh giới làm giảm độ chính xác ngay cả khi tỉ lệ nhiễu nhỏ | Cùng trực giác "lỗi ở ranh giới mới hại"; không có công thức ngưỡng |
| Part-dependent label noise (Xia và cs. 2020, NeurIPS) | Mô hình hóa nhiễu phụ thuộc mẫu cho học sâu | Không về ngưỡng |

**Hàm ý cho (B):** trực giác "nhiễu dồn ở ranh giới làm hỏng ngưỡng" đã được nêu (Menon 2018, Oyen 2022). Điều mới là **công thức định lượng** độ lệch qua tỉ lệ lỗi cục bộ α\*, β\* **tại ranh giới của mô hình đích**, và kiểm chứng định lượng trên dữ liệu thật (r = 0.94).

### 1.3. Ước lượng ngưỡng tối ưu (cut-point) và tốc độ hội tụ
| Công trình | Nội dung | Liên quan |
|---|---|---|
| Kim và Pollard 1990, Annals of Statistics, "Cube root asymptotics" | Định lý tổng quát cho ước lượng argmax có tốc độ n^(−1/3) | Công cụ chứng minh của (A) |
| Banerjee và McKeague 2007, Annals of Statistics, "Confidence sets for split points in decision trees" | Điểm chia của cây quyết định hội tụ n^(−1/3), giới hạn có phân phối Chernoff | Cùng dạng toán; ngưỡng Youden là một bài toán điểm chia |
| Thiele và Hirschfeld 2023, PLOS ONE | Khoảng tin cậy và cỡ mẫu cho cut-point tối ưu bằng mô phỏng; **không có công thức cỡ mẫu đóng**, không nhắc tốc độ n^(−1/3), không xét nhãn nhiễu | Cho thấy y văn ứng dụng chưa có công thức đóng; (A) lấp khoảng trống này, kể cả với nhãn thật (n_eff = n) |
| Các bài về Youden index ước lượng tham số / bán tham số (Fluss 2005, Yuan 2021…) | Ước lượng √n dưới giả định phân phối | Khác thiết lập (thực nghiệm phi tham số) |

### 1.4. ROC và ngưỡng khi chuẩn tham chiếu không hoàn hảo (thống kê chẩn đoán)
| Công trình | Nội dung | Liên quan |
|---|---|---|
| Bross 1954, Biometrics | Phân loại sai không phân biệt làm suy giảm hiệu số tỉ lệ theo hệ số (Se + Sp − 1) | Nguồn gốc của "đường Youden thu nhỏ theo 1 − α − β" trong dịch tễ học; nên trích |
| Vacek 1985, Biometrics; Torrance-Rynard và Walter 1997; các tổng quan về conditional dependence | Phụ thuộc có điều kiện giữa hai xét nghiệm làm lệch ước lượng độ nhạy, độ đặc hiệu | Tương tự (B) nhưng về ước lượng Se/Sp, không về chọn ngưỡng |
| Sun, Sang, Tian, Li 2025, arXiv 2502.08569 | ROC và Youden phi tham số khi chuẩn tham chiếu không hoàn hảo; **giả định độc lập có điều kiện và biết trước Se/Sp của chuẩn tham chiếu** | Gần với CATT về mục tiêu; khác ở chỗ cần biết Se/Sp và không phân tích khi vi phạm độc lập |
| Mô hình lớp ẩn không có chuẩn vàng (Henkelman 1990; Choi, Johnson 2006…) | Ước lượng ROC không cần chuẩn vàng, thường cần độc lập có điều kiện | Cùng giả định, khác mục tiêu |

### 1.5. Co-training và giám sát yếu
| Công trình | Nội dung | Liên quan |
|---|---|---|
| Blum và Mitchell 1998 | Co-training với hai góc nhìn độc lập có điều kiện | Nền của "góc nhìn khác" |
| Abney 2002; Balcan, Blum, Yang 2004 (expansion) | Nới lỏng giả định độc lập cho co-training | Không về ngưỡng |
| Ratner và cs. 2016, NeurIPS (data programming) | Số mẫu không nhãn cần thiết có **cùng bậc** với học có giám sát khi các hàm gán nhãn độc lập; hằng số kiểu 356/ε² | Cùng tinh thần "mẫu không nhãn đổi được sang mẫu có nhãn", nhưng về học tham số, tốc độ ε⁻², không phải ngưỡng Youden; không cho tỉ lệ đổi chính xác |
| Platanios và cs. 2014–2017 | Ước lượng độ chính xác từ dữ liệu không nhãn dựa vào sự đồng thuận của nhiều bộ phân loại (có xét phụ thuộc) | Có thể dùng làm hướng phát hiện thất bại không cần nhãn |

### 1.6. Ngưỡng và hiệu chuẩn khi chuyển viện không cần nhãn
| Công trình | Nội dung | Liên quan |
|---|---|---|
| Roschewitz và cs. 2023, Nature Communications (UPA) | Khớp phân phối điểm đích về nguồn | Đối thủ trực tiếp của CATT (đã so) |
| Flühmann, Bissoto, Hoang, Koch 2025, arXiv 2507.22776 | Ước lượng ma trận nhầm lẫn (PPV, NPV, độ nhạy, độ đặc hiệu) không cần nhãn trên X-quang ngực (CM-ATC, CM-DoC); suy giảm khi tỉ lệ bệnh thay đổi | Ước lượng hiệu năng, **không chọn lại ngưỡng**, không dùng mô hình thứ hai; nên trích như công trình liên quan mới |
| Patel và Beedala 2026, medRxiv (MIMIC-IV → eICU) | Hiệu chuẩn trôi khi chuyển viện; ngưỡng Youden giữ từ nguồn | Củng cố động cơ; không có phương pháp không cần nhãn |
| Vishwakarma và cs. 2025, ICML, "Rethinking confidence scores and thresholds in pseudolabeling-based SSL" | Chọn ngưỡng tin cậy cho nhãn giả trong học bán giám sát | Khác mục tiêu (lọc nhãn giả để huấn luyện, không chọn ngưỡng quyết định cuối) |

### 1.7. Kiểm tra "bị đi trước" (2025–2026)
Không tìm thấy công trình nào dùng mô hình thứ hai có cấu trúc khác (sinh lý, quy tắc, khái niệm) để chọn lại ngưỡng Youden của mô hình học sâu ở viện mới kèm luật cỡ mẫu hoặc công thức thiên lệch. Có một bản tóm tắt hội nghị JACC 2026 về "trôi điểm làm việc của AI-ECG phát hiện HFrEF" (không đọc được toàn văn), nên trích như bằng chứng vấn đề ngưỡng trôi trong AI-ECG nếu xác minh được nội dung.

## 2. Phát biểu tính mới nên dùng trong bài

| Nên viết | Không nên viết |
|---|---|
| "Building on the class-conditional noise invariance of balanced-error-optimal classifiers (Menon et al., 2015; Blanchard et al., 2016), we characterise the CATT threshold when the conditional-independence assumption fails, and derive its sampling behaviour." | "We propose a new theory of label noise." |
| "We show that, under conditional independence, CATT with n unlabelled samples has the same limiting distribution as a Youden threshold chosen with n_eff labelled samples, n_eff = …; this extends classical noisy-label sample-size results (Angluin and Laird, 1988) to cut-point estimation, which converges at the cube-root rate (Kim and Pollard, 1990)." | "The (1 − α − β)² factor is new." |
| "The bias is driven by the pseudo-labeller's error rates at the target model's decision boundary, quantifying the boundary-consistent noise intuition of Menon et al. (2018)." | "We are the first to show that boundary noise harms thresholds." |

## 3. Mức độ chặt của chứng minh

### 3.1. Công thức (B), Mệnh đề 1
- **Đã chặt** ở mức tổng thể: là phép lấy đạo hàm và quy đồng, không cần xấp xỉ. Điều kiện: fₖ liên tục, aₖ(t) đo được, q ∈ (0, 1). Khớp Monte Carlo đến nhiễu mẫu.
- Xấp xỉ bậc nhất (Γ/(D′ − Γ′)) là khai triển Taylor; cần nêu là xấp xỉ, kèm điều kiện Γ nhỏ.

### 3.2. Công thức (A), Mệnh đề 3: các bước cần viết chặt
1. **Thiết lập:** (fᵢ, ỹᵢ) độc lập cùng phân phối; J̃ₙ(t) = P̂(f ≥ t | ỹ = 1) − P̂(f ≥ t | ỹ = 0); t̂ = argmax J̃ₙ.
2. **Điều kiện (theo Kim–Pollard 1990, Định lý 1.1):**
   - t\* là điểm cực đại duy nhất, nằm trong miền xác định; J khả vi hai lần tại t\*, J″(t\*) < 0.
   - f₀, f₁ liên tục và dương tại t\*; điểm f không có giá trị trùng (phân phối liên tục).
   - Độc lập có điều kiện (ỹ ⊥ f | y), α + β < 1, q ∈ (0, 1).
   - Lớp hàm chỉ thị {1[f ≥ t]} có độ phức tạp VC hữu hạn (thỏa sẵn).
3. **Chỗ cần xử lý cẩn thận:** mẫu số q̂ là ngẫu nhiên. Phải chứng minh sai số q̂ − q = O_p(n^(−1/2)) chỉ ảnh hưởng ở bậc cao hơn n^(−1/3) (bổ đề ngắn).
4. **Kết quả chặt có thể phát biểu:** n^(1/3)(t̂ − t\*) hội tụ theo phân phối tới (4V/A²)^(1/3)·Z, với V = f(t\*)/[q(1 − q)], A = k|J″(t\*)|, k = π₀π₁(1 − α − β)/[q(1 − q)], Z theo phân phối Chernoff. Từ đó suy ra **tương đương phân phối**: CATT với n mẫu và Youden có nhãn với n_eff mẫu có cùng giới hạn.
5. **Phần chưa chặt:** công thức tổn thất **trung bình** (¼|J″|·Var(Z)·…) cần thêm điều kiện khả tích đều của n^(2/3)(t̂ − t\*)². Nên phát biểu kết quả theo phân phối (chặt) và để công thức trung bình là hệ quả "dưới điều kiện khả tích đều".

### 3.3. Kiểm chứng số đã có
- Mô phỏng: phân phối của ngưỡng CATT với n = 3.000 và của ngưỡng Youden có nhãn với n_eff = 1.127 trùng nhau (tỉ số độ lệch chuẩn 0.98; các phân vị 10–90% lệch ≤ 0.035).
- Số mũ: −0.64 (lý thuyết −0.67) theo n; −1.31 (−1.33) theo 1 − α − β.
- Dữ liệu thật: tỉ số quan sát/dự đoán tiến về 1 khi n_eff tăng ở cả Brugada và UCI Heart Disease (bước 12).

### 3.4. Nhờ ai đọc lại
- Người phù hợp: giảng viên thống kê hoặc xác suất có kinh nghiệm về lý thuyết tiệm cận (empirical processes, M-estimation), ví dụ ở khoa Toán ứng dụng của Bách khoa hoặc Viện Nghiên cứu cao cấp về Toán (VIASM).
- Câu hỏi cụ thể nên gửi kèm: (1) điều kiện Kim–Pollard ở mục 3.2 đã đủ chưa; (2) bổ đề về mẫu số ngẫu nhiên q̂; (3) có thể chứng minh khả tích đều để phát biểu tổn thất trung bình không.
- Nên gửi một bản phụ lục 2–3 trang (có thể soạn sẵn) thay vì toàn bộ bài.

## 4. Nguồn đã mở và đối chiếu
- Menon, van Rooyen, Natarajan 2018: https://users.cecs.anu.edu.au/~akmenon/papers/noisetron/noisetron.pdf
- Blanchard và cs. 2016 (arXiv 1303.1208): https://arxiv.org/pdf/1303.1208
- Banerjee và McKeague 2007 (arXiv 0708.1820): https://arxiv.org/pdf/0708.1820
- Ratner và cs. 2016 (arXiv 1605.07723): https://arxiv.org/pdf/1605.07723v1
- Sun và cs. 2025 (arXiv 2502.08569): https://arxiv.org/pdf/2502.08569
- Flühmann và cs. 2025 (arXiv 2507.22776): https://arxiv.org/html/2507.22776
- Oyen và cs. 2022 (arXiv 2206.01106): https://arxiv.org/abs/2206.01106v1
- Thiele và Hirschfeld 2023: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0279693
- Patel và Beedala 2026 (medRxiv): https://www.medrxiv.org/content/10.64898/2026.05.03.26352335v1.full.pdf
- Chỉ thấy qua kết quả tìm kiếm, chưa mở toàn văn: Angluin và Laird 1988; Natarajan và cs. 2013; Vacek 1985; Bross 1954; Vishwakarma và cs. 2025; tóm tắt JACC 2026. Cần đọc trước khi trích.
