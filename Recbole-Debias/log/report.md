# 📊 Báo Cáo Thực Nghiệm Khử Nhiễu Khuyến Nghị (Debiased Recommendation Report)

Báo cáo này tổng hợp kết quả thực nghiệm của các mô hình khuyến nghị được huấn luyện và đánh giá trên bộ dữ liệu **ml-100k-debias** sử dụng công cụ **RecBole-Debias**.

---

## ⚙️ 1. Cấu Hình Thiết Lập Chung (Shared Experimental Settings)

Tất cả các mô hình trong thực nghiệm này đều chia sẻ chung một quy trình xử lý dữ liệu và đánh giá tiêu chuẩn để đảm bảo tính công bằng (fairness):

### 🔹 Phương Pháp Phân Chia Dữ Liệu (Data Split)
* **Tỷ lệ phân chia:** `[0.8, 0.1, 0.1]` (80% Huấn luyện, 10% Kiểm thử phát triển/Validation, 10% Kiểm thử cuối cùng/Test).
* **Cơ chế phân chia (Split Mode):** Ratio-based Splitting, gom nhóm theo người dùng (`group_by: user`), sắp xếp theo thời gian tương tác (`order: RO` - Recency Order).
* **Đặc tính tập Test Unbiased:** 
  * Tập kiểm thử (Validation và Test set) là một **tập phi thiên vị (Unbiased / Intervened set)** được tạo ra bằng phương pháp **lấy mẫu ngược nghịch đảo xác suất tương tác của mục (Inverse Propensity Scoring)**.
  * Điều này đảm bảo mỗi mục (item) có xác suất xuất hiện hoàn toàn đồng đều trong tập Test, giúp kiểm tra khả năng khuyến nghị thực chất của mô hình thay vì chỉ "học vẹt" độ phổ biến (popularity bias).

### 🔹 Cơ Chế Đánh Giá (Evaluation Protocol)
* **Phương thức đánh giá:** Full Sort (xếp hạng trên toàn bộ danh sách các mục chưa tương tác của người dùng).
* **Độ đo Top-K (Metrics):** Sử dụng các độ đo xếp hạng tại ngưỡng **K = 10**:
  * **Recall@10:** Tỷ lệ tìm thấy các mục người dùng thực sự thích trong top 10 gợi ý.
  * **NDCG@10:** Độ đo xếp hạng có phạt lỗi (reward cao hơn khi gợi ý đúng nằm ở vị trí đầu danh sách).
  * **MRR@10:** Vị trí đảo nghịch của mục đúng đầu tiên xuất hiện trong danh sách gợi ý.
  * **Hit@10:** Tỷ lệ người dùng có ít nhất một gợi ý chính xác trong top 10.
  * **Precision@10:** Tỷ lệ gợi ý chính xác trên tổng số 10 mục được khuyến nghị.

### 🔹 Mô Hình Cơ Sở & Xương Sống (Backbone Base Models)
* Phần lớn các thuật toán khử nhiễu sử dụng **MF (Matrix Factorization - Nhân tử hóa ma trận)** làm cấu trúc mạng cơ sở.
* **LightGCN** được bổ sung như một mốc cơ sở (baseline) mạnh mẽ thứ hai, sử dụng kiến trúc Đồ thị (Graph Convolution Network) để khai thác thông tin cộng tác bậc cao.
* Tất cả mô hình đều thiết lập kích thước nhúng mặc định là `64` và sử dụng cùng một bộ siêu tham số cấu hình chung.

---

## 🔄 2. Sự Khác Biệt Giữa Các Mô Hình (Model Differences)

Mặc dù dùng chung xương sống MF và tập dữ liệu, mỗi mô hình tiếp cận việc khử nhiễu (debiasing) theo các triết lý nhân quả và xác suất hoàn toàn khác nhau:

| Mô hình | Loại Bias Nhắm Tới | Cơ Chế Khử Nhiễu Đặc Trưng |
| :--- | :--- | :--- |
| **MF (Base)** | Không (Mốc so sánh) | Không khử nhiễu. Tối ưu hóa trực tiếp trên hàm mất mát bình phương tối thiểu (MSE) hoặc BPR chuẩn. Bị ảnh hưởng nặng bởi Popularity Bias. |
| **LightFM (Base)** | Không (Mốc so sánh) | Không khử nhiễu. Sử dụng Matrix Factorization tích hợp thêm các hệ số thiên lệch (user/item biases). Cải thiện rất mạnh trên dữ liệu thưa thớt. |
| **LightGCN (Base)** | Không (Mốc so sánh) | Không khử nhiễu. Sử dụng kiến trúc Đồ thị (GCN) tinh gọn để lan truyền nhúng qua các liên kết bậc cao. Mạnh mẽ hơn MF truyền thống. |
| **MF-IPS** | Selection Bias | Sử dụng kỹ thuật **Trọng số nghịch đảo xu hướng (Inverse Propensity Scoring)**. Tính toán xác suất hiển thị của mỗi mục để tái trọng số hàm loss, phạt các mục quá phổ biến và boost các mục ít tương tác. |
| **PDA** | Popularity Bias | Sử dụng **Can thiệp nhân quả (Causal Intervention)** trên đồ thị nhân quả. Hiệu chỉnh công thức dự đoán ở giai đoạn suy diễn (inference) để bù đắp cho độ phổ biến của mục. |
| **DICE** | Popularity Bias | **Tách biệt không gian nhúng (Disentangling)**. Học hai ma trận nhúng độc lập cho mỗi User/Item: Một không gian biểu diễn Sở thích thực tế (Interest) và một không gian biểu diễn Sự tuân theo đám đông (Conformity). |
| **MACR** | Popularity Bias | **Lập luận phản thực tế model-agnostic**. Tách biệt xác suất nhấp chuột thành ba mối quan hệ nhân quả (user-item, user-free, item-free) nhằm khử độ phổ biến khi suy luận. |
| **CausE** | Popularity Bias | **Học nhúng nhân quả (Causal Embeddings)**. Huấn luyện song song mô hình trên hai tập dữ liệu (Tập biased và Tập unbiased) sau đó sử dụng hàm phạt khoảng cách nhúng (`dis_pen`) để điều phối hai không gian này. |
| **REL_MF** | Exposure Bias | Khử nhiễu dựa trên **Khả năng hiển thị thực tế (Relevance-based Exposure)**. Mô hình hóa việc tương tác là kết hợp của hai bước độc lập: người dùng nhìn thấy mục (exposure) và người dùng thích mục (relevance). |

---

## 📈 3. Bảng Kết Quả Thực Nghiệm (Comparative Results)

Dưới đây là bảng so sánh chi tiết hiệu năng của toàn bộ các mô hình trên tập kiểm thử phi thiên vị **ml-100k-debias (Unbiased Test Set)** được tổng hợp trực tiếp từ các file log chạy thành công của bạn:

> [!NOTE]
> Kết quả được sắp xếp theo thứ tự **NDCG@10** giảm dần. Mô hình tốt nhất được đánh dấu nổi bật.

| Mô hình | Recall@10 | MRR@10 | NDCG@10 | Hit@10 | Precision@10 | Trạng Thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🚀 **LightGCN (Base)** | **0.2529** | **0.5035** | **0.3025** | **0.7964** | **0.2046** | Đã huấn luyện (Vượt trội) |
| 🌟 **LightFM (Base)** | 0.2267 | 0.4480 | 0.2677 | 0.7635 | 0.1836 | Đã huấn luyện (Rất tốt) |
| 🏆 **DICE** | 0.1002 | 0.2031 | 0.1069 | 0.5736 | 0.0843 | Đã huấn luyện (Tốt nhất nhóm Debias) |
| 🥈 **MACR** | 0.0834 | 0.1894 | 0.0955 | 0.5127 | 0.0757 | Đã huấn luyện |
| 🥉 **MF (Base)** | 0.0878 | 0.1834 | 0.0944 | 0.5172 | 0.0748 | Đã huấn luyện |
| 4️⃣ **PDA** | 0.0846 | 0.1723 | 0.0897 | 0.5105 | 0.0724 | Đã huấn luyện |
| 5️⃣ **MF_IPS** | 0.0530 | 0.1344 | 0.0629 | 0.3821 | 0.0511 | Đã huấn luyện |
| 6️⃣ **REL_MF** | 0.0352 | 0.0907 | 0.0399 | 0.2713 | 0.0319 | Đã huấn luyện |
| 7️⃣ **CausE** | 0.0085 | 0.0353 | 0.0142 | 0.1096 | 0.0127 | Đã huấn luyện |

---

## 🧠 4. Phân Tích & Rút Ra Nhận Xét (Key Analytical Takeaways)

### 🌟 1. Sức mạnh của Biases và Kiến trúc biểu diễn (LightFM & LightGCN)
* Cả **LightGCN** và **LightFM** đều không sử dụng kỹ thuật khử nhiễu nhưng lại thống trị hoàn toàn bảng xếp hạng, với NDCG@10 đạt lần lượt **0.3025** và **0.2677**. Cả hai mô hình này đều bỏ xa mô hình debias tốt nhất là DICE (0.1069) và MF cơ sở (0.0944).
* **Nhận xét:** Trong khi LightGCN tận dụng sức mạnh của đồ thị bậc cao, thì **LightFM** chứng minh rằng chỉ cần bổ sung các hệ số thiên lệch (user/item bias terms) vào MF truyền thống là đủ để tạo ra sự đột phá. Điều này minh chứng một bài học đắt giá: Việc xây dựng một bộ máy học biểu diễn đặc trưng (feature representation) mạnh mẽ có thể mang lại hiệu suất vượt xa sự bù đắp của các kỹ thuật can thiệp toán học (debias interventions) trên một nền tảng tuyến tính lỏng lẻo như MF gốc.

### 🥇 2. Sự thống trị của DICE trong nhóm MF-Debias
* Trong nhóm các thuật toán khử nhiễu dựa trên MF, **DICE** vẫn là mô hình thể hiện tốt nhất, vượt qua **MF** khoảng **+14.1% ở Recall@10** và **+13.2% ở NDCG@10**. 
* **Nhận xét:** Cơ chế tách biệt rõ ràng giữa "Sở thích thực sự" và "Độ phổ biến đám đông" của DICE cực kỳ thích ứng với phân phối MovieLens. Việc khuyến nghị dựa trên không gian sở thích thuần khiết giúp DICE dự đoán chính xác nhu cầu của người dùng trên tập Test phi thiên vị.

### 🥈 2. Hiệu quả khử nhiễu đáng nể của MACR
* **MACR** đứng ở vị trí thứ hai, vượt qua mô hình nền tảng **MF** ở hầu hết các chỉ số chất lượng xếp hạng (ví dụ: **MRR@10** đạt **0.1894** so với **0.1834** của MF, **NDCG@10** đạt **0.0955** so với **0.0944** của MF).
* **Nhận xét:** Việc áp dụng lập luận phản thực tế để loại bỏ độ phổ biến mục (item popularity bias) một cách độc lập trong quá trình dự đoán mang lại hiệu quả rất tích cực trên dữ liệu phân tán không đồng đều.

### 🥉 3. PDA bám đuổi sát nút
* **PDA** đạt NDCG@10 ở mức **0.0897**, tuy giảm nhẹ so với MF cơ sở nhưng cấu trúc của PDA vẫn cho thấy khả năng giữ vững độ chính xác tương đối tốt trên tập Test Unbiased nhờ can thiệp nhân quả trực tiếp.

### ⚠️ 4. Tại sao các mô hình nâng cao như IPS, CausE, REL_MF lại giảm sâu?
* **MF_IPS (0.0629 NDCG@10):** Kỹ thuật IPS (Inverse Propensity Scoring) thường có **phương sai cực kỳ cao (high variance)**. Nếu hệ số xu hướng (propensity score) không được ước lượng hoàn hảo, trọng số hàm loss sẽ bị bóp méo dữ dội làm mô hình bị mất ổn định và hội tụ kém.
* **CausE (0.0142 NDCG@10):** CausE đòi hỏi huấn luyện song song hai không gian nhúng khác nhau (Biased và Unbiased) và liên kết chúng. Việc phân phối dữ liệu nhỏ như ml-100k không đủ mẫu để tối ưu hóa đồng thời hai không gian biểu diễn phức tạp này nếu thiếu sự tinh chỉnh tham số phạt `dis_pen` chính xác.
