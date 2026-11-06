# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603770
- **Họ và tên:** Bùi Ngọc Hùng
- **SEED:** 178
- **Challenge:** #5 (challenge index = 4 trong JSON)

---

## 1. Bằng chứng

Các tệp minh chứng được lưu trong thư mục `report/individual/`.

| Tệp | Nội dung | Trạng thái |
|---|---|---|
| [day01_lab.ipynb](day01_lab.ipynb) | Notebook thực hiện bài lab và lưu kết quả | ☑ |
| [lab1_report.json](lab1_report.json) | Kết quả thực nghiệm và dự đoán ban đầu | ☑ |
| [hand_calc.jpg](hand_calc.jpg) | Ảnh ghi lại các phép tính thủ công | ☑ |

**Kết quả kiểm tra tự động:** 15/15 tests passed.

Các nhóm chức năng đã hoàn thành:

- Vectorization, sigmoid, predict_proba và Logistic Regression.
- Xây dựng và huấn luyện MLP.
- Cosine similarity, top-k similarity và kNN.
- Prompt, parse label và gọi LLM API.
- Các phép tính tay về Accuracy, Sigmoid, F1, Cosine Similarity và chi phí LLM.

---

## 2. Kết quả chính

### 2.1. Bảng so sánh mô hình

| Mô hình | F1 test | F1 SUB | p95 (ms/tin) | USD/1k tin |
|---|---:|---:|---:|---:|
| BoW + LogReg | 0.918 | 0.974 | 0.065 | 0.00 |
| BoW + MLP | 0.928 | 0.974 | 0.269 | 0.00 |
| MiniLM + kNN | 0.868 | 0.950 | 52.114 | 0.00 |
| MiniLM + LogReg | 0.959 | 1.000 | 31.368 | 0.00 |
| MiniLM + MLP head | 0.941 | 1.000 | 32.627 | 0.00 |
| LLM nhỏ zero-shot | 0.889 | 0.889 | 2389.609 | 0.00 |
| LLM lớn zero-shot | 0.952 | 0.952 | 2403.436 | 0.00 |
| LLM nhỏ few-shot | 0.930 | 0.930 | 2449.495 | 0.00 |

### 2.2. Nhận xét

- **BoW + LogReg:** F1 test đạt 0.918, F1 SUB đạt 0.974. Đây là mô hình có tốc độ suy luận nhanh nhất, p95 khoảng 0.065 ms/tin. Precision đạt 1.000 nhưng Recall khoảng 0.849, cho thấy mô hình có thể bỏ sót một số tin nhắn spam.
- **BoW + MLP:** F1 test đạt 0.928, cao hơn BoW + LogReg, nhưng F1 SUB không thay đổi.
- **MiniLM + kNN:** F1 test đạt 0.868, p95 khoảng 52.114 ms/tin, không đáp ứng yêu cầu p95 dưới 50 ms.
- **MiniLM + LogReg:** F1 test đạt 0.959, F1 SUB đạt 1.000 và p95 khoảng 31.368 ms/tin. Đây là mô hình có chất lượng tốt nhất trong nhóm ML/DL/Embedding đã thử nghiệm.
- **MiniLM + MLP head:** F1 test đạt 0.941, F1 SUB đạt 1.000. Tuy nhiên, p95 cao hơn MiniLM + LogReg.
- **LLM:** LLM lớn zero-shot đạt F1 0.952, cao hơn LLM nhỏ zero-shot 0.889. Tuy nhiên, p95 của các mô hình LLM khoảng 2.4 giây/tin, không đáp ứng yêu cầu xử lý nhanh.

**Nhận xét về chi phí:** Giá trị USD/1k tin trong JSON đều được ghi là 0.00. Đây là giá trị được lưu trong báo cáo thực nghiệm, không đủ căn cứ để kết luận việc sử dụng API thực tế hoàn toàn miễn phí.

---

## 3. Dự đoán và kết quả thực tế

### 3.1. Dự đoán F1 của Logistic Regression

- **Dự đoán:** F1 khoảng 0.85.
- **Thực tế:** BoW + LogReg đạt F1 = 0.918.
- **Nhận xét:** Kết quả thực tế cao hơn dự đoán ban đầu. Điều này cho thấy Logistic Regression kết hợp với Bag of Words có khả năng phân loại tốt trên tập dữ liệu của bài lab.

### 3.2. So sánh LLM nhỏ và LLM lớn

- **Dự đoán:** Model lớn có F1 cao hơn model nhỏ khoảng 2–5%.
- **Thực tế:** LLM nhỏ zero-shot đạt F1 = 0.889, LLM lớn zero-shot đạt F1 = 0.952.
- **Chênh lệch:** 0.952 - 0.889 = 0.063, tương đương 6.3 điểm phần trăm.
- **Nhận xét:** Model lớn cải thiện F1 nhiều hơn dự đoán ban đầu. Tuy nhiên, độ trễ của cả hai mô hình đều khoảng 2.4 giây/tin, không phù hợp với yêu cầu xử lý nhanh.

### 3.3. Thử nghiệm bỏ ReLU

- **Dự đoán:** Bỏ ReLU có thể làm F1 giảm vì mạng mất khả năng học các quan hệ phi tuyến.
- **Thực tế:** Kết quả thử nghiệm bỏ ReLU đạt F1 khoảng 0.934, so với 0.928 của MLP ban đầu.
- **Nhận xét:** Kết quả không giảm như dự đoán mà tăng nhẹ. Tuy nhiên, việc bỏ ReLU khiến các lớp tuyến tính liên tiếp không còn tạo được khả năng biểu diễn phi tuyến. Kết quả này chỉ phản ánh dữ liệu và cấu hình đang thử nghiệm, chưa đủ để kết luận bỏ ReLU luôn tốt hơn.

---

## 4. Phân tích 50 case khó

### 4.1. Kết quả stress test

Kết quả đánh giá trên 50 trường hợp khó:

| Mô hình | F1 | Accuracy |
|---|---:|---:|
| BoW + LogReg | 0.378 | 0.54 |
| BoW + MLP | 0.421 | 0.56 |
| Embedding + kNN | 0.750 | 0.72 |
| Embedding + LogReg | 0.612 | 0.62 |
| Embedding + MLP | 0.600 | 0.60 |
| LLM nhỏ zero-shot | 0.918 | 0.90 |
| LLM nhỏ defensive prompt | 0.935 | 0.92 |

### 4.2. Phân tích theo nhóm dữ liệu

Accuracy của BoW + LogReg trên các nhóm case:

| Nhóm | Accuracy |
|---|---:|
| Tưởng spam nhưng không phải | 100% |
| Thao túng / lừa đảo | 12% |
| Spam giọng tự nhiên | 20% |
| Né từ khóa | 17% |
| Prompt injection | 67% |
| Tin giao dịch hợp lệ | 83% |
| Nói về spam | 100% |

### 4.3. Nhận xét và nguyên nhân

BoW + LogReg giảm mạnh từ F1 SUB = 0.974 xuống F1 = 0.378 trên tập 50 case khó. Điều này cho thấy mô hình gặp khó khăn khi dữ liệu có cách diễn đạt khác biệt so với dữ liệu đã học.

Các nhóm khó nhất đối với BoW + LogReg là:

- Thao túng / lừa đảo: Accuracy chỉ 12%.
- Spam giọng tự nhiên: Accuracy 20%.
- Né từ khóa: Accuracy 17%.

Một số tình huống có thể gây nhầm lẫn:

- **Ham có từ khóa giống spam:** Ví dụ “Claim your seat early tomorrow, the lecture hall fills up fast”. Từ `Claim` có thể khiến mô hình dựa trên từ khóa nghiêng về spam, mặc dù ngữ cảnh thực tế là lời nhắc giữ chỗ trong giảng đường.
- **Ham nhắc đến khuyến mãi:** Ví dụ “I got 50% off at the shoe store today, what a find! Want me to grab you a pair?”. Cụm `50% off` thường xuất hiện trong quảng cáo, nhưng đây là tin nhắn cá nhân chia sẻ việc mua sắm.

Hai ví dụ trên chỉ dùng để minh họa tình huống ham có từ khóa giống spam, không khẳng định là lỗi dự đoán cụ thể của mô hình nếu chưa đối chiếu output từng tin nhắn.

Nguyên nhân chính là BoW chủ yếu dựa vào sự xuất hiện của từ khóa, chưa hiểu đầy đủ ngữ cảnh. Embedding biểu diễn văn bản theo ngữ nghĩa, còn LLM có khả năng phân tích ngữ cảnh và nội dung câu tốt hơn trong những trường hợp phức tạp.

LLM nhỏ với defensive prompt đạt F1 = 0.935, cao hơn zero-shot là 0.918. Accuracy cũng tăng từ 0.90 lên 0.92. Điều này cho thấy việc bổ sung hướng dẫn phòng thủ trong prompt có thể cải thiện kết quả trên tập case khó.

### 4.4. Kết luận

Trong tập stress test, LLM nhỏ defensive prompt đạt F1 cao nhất trong các mô hình được thử nghiệm. Embedding + kNN cũng cải thiện đáng kể so với BoW. Tuy nhiên, chất lượng phân loại cần được đánh giá cùng độ trễ và chi phí trước khi lựa chọn mô hình triển khai.

---

## 5. Thử thách cá nhân

- **SEED:** 178.
- **Đề thử thách:** Challenge #5 – Embedding (challenge index = 4 trong JSON).

Thử thách yêu cầu tự tạo 4 tin nhắn, bao gồm:

- 2 tin nhắn spam cố tình né từ khóa bằng cách thay đổi ký tự và chèn khoảng trắng.
- 2 tin nhắn ham có chứa các từ `free` hoặc `win` nhưng nội dung thực tế là giao tiếp bình thường.

Sau đó, sử dụng BoW + LogReg và Encoder + LogReg để dự đoán, so sánh kết quả và giải thích sự khác nhau.

### 5.1. Kết quả dự đoán thực tế

| STT | Nội dung thử thách | Nhãn thật | BoW + LogReg | Encoder + LogReg |
|---|---|---|---|---|
| 1 | Spam sử dụng leet-speak: thay chữ bằng số | SPAM | HAM ❌ | SPAM ✅ |
| 2 | Spam chèn khoảng trắng vào từ khóa | SPAM | HAM ❌ | SPAM ✅ |
| 3 | Ham có chứa từ `free` | HAM | HAM ✅ | HAM ✅ |
| 4 | Ham có chứa từ `win` | HAM | HAM ✅ | HAM ✅ |

### 5.2. Tổng hợp kết quả

| Mô hình | Số dự đoán đúng | Accuracy |
|---|---:|---:|
| BoW + LogReg | 2/4 | 50% |
| Encoder + LogReg | 4/4 | 100% |

### 5.3. Phân tích

- **BoW + LogReg:** Dự đoán sai cả hai tin nhắn spam. Khi ký tự bị thay đổi hoặc từ khóa bị chèn khoảng trắng, các đặc trưng từ vựng mà mô hình đã học có thể không còn khớp với văn bản đầu vào.
- **Encoder + LogReg:** Dự đoán đúng cả bốn tin nhắn. Encoder biểu diễn văn bản theo ngữ nghĩa tốt hơn, giúp mô hình nhận diện nội dung spam ngay cả khi cách viết từ bị biến đổi.
- **Đối với hai tin nhắn ham:** Cả hai mô hình đều phân loại đúng. Điều này cho thấy không nên chỉ dựa vào sự xuất hiện của các từ như `free` hoặc `win` để kết luận một tin nhắn là spam.

**Kết luận:** Trong thử thách gồm bốn tin nhắn tự tạo, Encoder + LogReg đạt 4/4 dự đoán đúng, trong khi BoW + LogReg đạt 2/4. Kết quả cho thấy Encoder có lợi thế trong những trường hợp văn bản cố tình né từ khóa. Tuy nhiên, số lượng mẫu thử còn nhỏ nên chưa đủ để kết luận mô hình luôn tốt hơn trên dữ liệu thực tế.

---

## 6. Phương án cho sản phẩm thật

Với yêu cầu p95 < 50 ms/tin và 10 triệu SMS/ngày, em chọn **MiniLM + LogReg** làm mô hình phân loại chính.

### 6.1. Lý do lựa chọn

- F1 test = 0.959.
- F1 SUB = 1.000.
- p95 = 31.368 ms/tin, đáp ứng SLA dưới 50 ms trong phép đo của bài lab.
- Throughput đo được khoảng 58.5 tin/giây.

### 6.2. Khả năng đáp ứng tải

10 triệu SMS/ngày tương đương trung bình:

\[
\frac{10,000,000}{86,400} \approx 115.74 \text{ tin/giây}
\]

MiniLM + LogReg đạt throughput đo được khoảng 58.5 tin/giây trong thí nghiệm.

Như vậy, một instance theo kết quả benchmark hiện tại chưa đủ để đáp ứng mức tải trung bình 115.74 tin/giây.

Cần triển khai nhiều instance chạy song song, sử dụng load balancer và thực hiện benchmark dưới tải thực tế để xác định số lượng máy cần thiết. Đồng thời cần tính đến tải cao điểm, tài nguyên CPU/RAM và khả năng mở rộng khi lượng SMS tăng.

### 6.3. Đánh đổi và chi phí LLM

Nếu hệ thống ưu tiên tốc độ và tiết kiệm tài nguyên hơn, BoW + LogReg là lựa chọn thay thế vì có p95 chỉ 0.065 ms/tin và F1 SUB đạt 0.974.

Không nên đưa LLM vào luồng xử lý chính vì p95 khoảng 2.4 giây/tin, không đáp ứng SLA.

Theo phép tính chi phí thủ công trong bài lab, nếu xử lý 1 triệu SMS/ngày với chi phí khoảng 10.50 USD/ngày thì với 10 triệu SMS/ngày:

- Chi phí mỗi ngày: 10.50 × 10 = 105 USD.
- Chi phí trong 30 ngày: 105 × 30 = **3,150 USD/tháng**.

Đây là ước tính theo giả định chi phí tăng tuyến tính với số lượng tin nhắn và mức sử dụng token. Chi phí thực tế có thể thay đổi tùy model, độ dài tin nhắn và giá API tại thời điểm sử dụng.

### 6.4. Kiến trúc đề xuất – Hybrid Router

Em đề xuất kết hợp MiniLM + LogReg với LLM theo cơ chế confidence-based router:

1. **Tiếp nhận:** SMS Gateway nhận tin nhắn và thực hiện tiền xử lý.
2. **Phân loại chính:** MiniLM tạo embedding, sau đó Logistic Regression dự đoán nhãn và xác suất spam.
3. **Tin có độ tin cậy cao:** Nếu xác suất dưới 0.3 hoặc trên 0.7, hệ thống sử dụng kết quả của mô hình chính.
4. **Tin chưa chắc chắn:** Nếu xác suất nằm trong khoảng [0.3, 0.7], hệ thống chuyển tin nhắn sang LLM để phân tích thêm ngữ cảnh.
5. **Trả kết quả:** Kết hợp kết quả phân tích và đưa ra nhãn HAM hoặc SPAM cuối cùng.

Ngưỡng 0.3–0.7 là giá trị đề xuất ban đầu, cần được hiệu chỉnh bằng tập validation trước khi triển khai.

Thiết kế này giúp phần lớn tin nhắn được xử lý bằng mô hình có tốc độ nhanh, trong khi LLM chỉ xử lý những trường hợp khó. Nhờ đó, hệ thống có thể giảm số lần gọi API, tiết kiệm chi phí và cân bằng giữa chất lượng phân loại với tốc độ xử lý.

Tuy nhiên, cần kiểm chứng tỷ lệ tin nhắn được chuyển sang LLM, độ trễ tổng thể và chất lượng phân loại trước khi áp dụng thực tế.

**Kết luận:** MiniLM + LogReg là lựa chọn cân bằng tốt giữa chất lượng và độ trễ trong các mô hình đã thử nghiệm. Kết hợp với LLM qua Router có thể cải thiện khả năng xử lý các trường hợp khó, nhưng cần tiếp tục kiểm thử tải và đánh giá chi phí trước khi triển khai chính thức.

---

## 7. Prediction Log, AI Usage Log và điều đã học

### 7.1. Prediction Log – Dự đoán trước khi chạy thí nghiệm

Các dự đoán ban đầu được ghi trong `lab1_report.json` với SEED = 178:

| Thời gian | Nội dung dự đoán |
|---|---|
| 13:49:44 | F1 khoảng 0.85; dự đoán Precision cao hơn Recall |
| 13:49:47 | lr = 0.001 giảm loss chậm; lr = 50 dễ tăng hoặc dao động; lr = 0.5 giảm ổn định |
| 13:50:00 | Validation loss có thể tăng trở lại; MLP có thể hơn Logistic Regression một chút |
| 13:50:07 | Bỏ ReLU có thể làm F1 giảm nhẹ |
| 13:51:59 | Kết quả gần nhất trong embedding search có thể là spam |
| 13:52:50 | Encoder + LogReg dự kiến đoán đúng nhiều hơn BoW |
| 13:52:59 | Model lớn dự kiến có F1 cao hơn model nhỏ khoảng 2–5%; model lớn có thể đắt hơn |
| 13:54:07 | Few-shot có thể giúp F1 tăng nhẹ nhưng làm tăng chi phí token |
| 13:55:06 | BoW + LogReg và MiniLM + LogReg có thể nằm trên Pareto front |
| 13:55:07 | BoW dự kiến giảm mạnh trên 50 case khó; embedding và LLM có thể tốt hơn |

### 7.2. AI Usage Log – Nhật ký sử dụng AI

Trong quá trình thực hiện bài lab, em sử dụng ChatGPT để hỗ trợ giải thích kiến thức ML/DL, tìm hiểu lỗi trong quá trình triển khai code và hỗ trợ phân tích kết quả thực nghiệm.

| Công cụ | Mục đích sử dụng | Cách kiểm chứng |
|---|---|---|
| ChatGPT | Giải thích thuật toán và các khái niệm ML/DL | Đối chiếu với nội dung bài học và kết quả chạy notebook |
| ChatGPT | Hỗ trợ tìm hiểu lỗi code và cách triển khai | Chạy lại code và kiểm tra kết quả thực tế |
| ChatGPT | Hỗ trợ phân tích kết quả và trình bày báo cáo | Đối chiếu số liệu với `lab1_report.json` và output notebook |

**Vai trò cá nhân:** Em trực tiếp thực hiện các thí nghiệm, kiểm tra kết quả, đánh giá các mô hình và quyết định nội dung cuối cùng của báo cáo. Các gợi ý từ AI chỉ được sử dụng như tài liệu hỗ trợ và không thay thế kết quả thực nghiệm.

### 7.3. Điều em học được

Qua bài lab, em hiểu rõ hơn sự khác nhau giữa các phương pháp phân loại văn bản:

- **BoW + Logistic Regression:** Đơn giản, tốc độ cao nhưng phụ thuộc nhiều vào đặc trưng từ vựng.
- **MLP:** Có thể học các quan hệ phi tuyến nhờ activation function, nhưng mô hình phức tạp hơn không phải lúc nào cũng cho kết quả tốt hơn.
- **Embedding:** Biểu diễn văn bản theo ngữ nghĩa, giúp xử lý tốt hơn một số trường hợp thay đổi cách diễn đạt.
- **LLM:** Có khả năng phân tích ngữ cảnh và xử lý dữ liệu khó, nhưng độ trễ cao.
- **Few-shot và prompt engineering:** Có thể cải thiện kết quả nhưng cần cân nhắc thêm chi phí và thời gian phản hồi.

Điều em dự đoán chưa chính xác nhất là việc bỏ ReLU sẽ làm F1 giảm. Kết quả thực tế lại tăng nhẹ từ 0.928 lên khoảng 0.934. Qua đó, em hiểu rằng kết quả thực nghiệm có thể khác với giả thuyết ban đầu và cần được kiểm chứng bằng dữ liệu.

Em cũng nhận ra rằng không nên chỉ đánh giá mô hình bằng F1. Khi lựa chọn mô hình cho sản phẩm thực tế, cần xem xét đồng thời chất lượng phân loại, độ trễ, throughput, chi phí và khả năng mở rộng hệ thống.

**Tổng kết:** Qua Day 01, em đã thực hành xây dựng và đánh giá các mô hình Machine Learning, Deep Learning, Embedding và LLM cho bài toán phân loại SMS spam. Quan trọng hơn, em hiểu cách so sánh mô hình dựa trên kết quả thực nghiệm và các yêu cầu triển khai thực tế.
