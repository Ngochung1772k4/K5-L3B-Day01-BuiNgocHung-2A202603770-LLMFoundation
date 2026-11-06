# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603770
- **Họ và tên:** Bùi Ngọc Hùng
- **SEED:** 178

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [day01_lab.ipynb](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [lab1_report.json](lab1_report.json) — kết quả báo cáo | ☑ |
| [hand_calc.jpg](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Bảng so sánh các mô hình trên tập test và tập SUB:

| Mô hình | F1 test | F1 SUB | p95 (ms/tin) | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.918 | 0.974 | 0.065 | 0.00 |
| BoW + MLP | 0.928 | 0.974 | 0.269 | 0.00 |
| MiniLM + kNN | 0.868 | 0.950 | 52.114 | 0.00 |
| MiniLM + LogReg | 0.959 | 1.000 | 31.368 | 0.00 |
| MiniLM + MLP head | 0.941 | 1.000 | 32.627 | 0.00 |
| LLM nhỏ zero-shot | 0.889 | 0.889 | 2389.609 | 0.00 |
| LLM lớn zero-shot | 0.952 | 0.952 | 2403.436 | 0.00 |
| LLM nhỏ few-shot | 0.930 | 0.930 | 2449.495 | 0.00 |

MiniLM + LogReg đạt F1 test cao nhất trong nhóm mô hình ML/DL/Embedding, với 0.959 và F1 SUB = 1.000. BoW + LogReg có độ trễ thấp nhất, chỉ 0.065 ms/tin.

LLM lớn đạt F1 = 0.952, cao hơn LLM nhỏ zero-shot là 0.889. Tuy nhiên, độ trễ của LLM khoảng 2.4 giây/tin, cao hơn nhiều so với các mô hình còn lại.

*Lưu ý: Giá trị USD/1k tin trong JSON đều bằng 0.00. Đây là giá trị được ghi trong báo cáo thực nghiệm, không đủ căn cứ để kết luận API thực tế hoàn toàn miễn phí.*

## 3. Dự đoán và thực tế

### 3.1. Dự đoán F1 của Logistic Regression

- Dự đoán: F1 khoảng 0.85.
- Thực tế: BoW + LogReg đạt F1 = 0.918.
- Nhận xét: Kết quả thực tế cao hơn dự đoán. Mô hình đạt precision = 1.00 nhưng recall = 0.849, tức là vẫn bỏ sót một số tin nhắn spam.

### 3.2. So sánh LLM nhỏ và LLM lớn

- Dự đoán: Model lớn có F1 cao hơn model nhỏ khoảng 2–5%.
- Thực tế: LLM nhỏ đạt 0.889, LLM lớn đạt 0.952.
- Chênh lệch: 0.952 - 0.889 = 0.063, tương đương khoảng 6.35 điểm phần trăm.
- Nhận xét: Model lớn cải thiện F1 nhiều hơn dự đoán ban đầu. Tuy nhiên, cả hai đều có độ trễ cao.

### 3.3. Thử nghiệm bỏ ReLU

- Dự đoán: Bỏ ReLU có thể làm F1 giảm nhẹ vì mạng mất khả năng học quan hệ phi tuyến.
- Thực tế: Kết quả thử nghiệm bỏ ReLU đạt F1 khoảng 0.934, so với 0.928 của MLP ban đầu.
- Nhận xét: Kết quả không giảm như dự đoán. Khi không có ReLU, hai lớp tuyến tính kết hợp vẫn tương đương một phép biến đổi tuyến tính, nên không tạo thêm khả năng biểu diễn phi tuyến.

## 4. Phân tích 50 case khó

Kết quả stress test:

| Mô hình | F1 | Accuracy |
| --- | ---: | ---: |
| BoW + LogReg | 0.378 | 0.54 |
| BoW + MLP | 0.421 | 0.56 |
| Embedding + kNN | 0.750 | 0.72 |
| Embedding + LogReg | 0.612 | 0.62 |
| Embedding + MLP | 0.600 | 0.60 |
| LLM nhỏ zero-shot | 0.918 | 0.90 |
| LLM nhỏ defensive prompt | 0.935 | 0.92 |

BoW + LogReg giảm mạnh nhất: F1 từ 0.974 trên tập SUB xuống 0.378 trên 50 case khó.

Hai tình huống dễ gây nhầm lẫn:

- **Ham có từ khóa giống spam:** Ví dụ tin nhắn “Claim your seat early tomorrow, the lecture hall fills up fast”. Từ `Claim` có thể khiến mô hình dựa trên từ khóa nghiêng về spam, mặc dù ngữ cảnh thực tế là lời nhắc giữ chỗ trong giảng đường.
- **Ham nhắc đến khuyến mãi:** Ví dụ “I got 50% off at the shoe store today, what a find! Want me to grab you a pair?”. Cụm `50% off` thường xuất hiện trong quảng cáo, nhưng đây là tin nhắn cá nhân chia sẻ việc mua sắm.

Nguyên nhân là BoW chủ yếu dựa vào sự xuất hiện của từ khóa, chưa hiểu đầy đủ ngữ cảnh. Embedding và LLM có lợi thế hơn khi cần phân tích ý nghĩa của cả câu.

LLM nhỏ với defensive prompt đạt F1 = 0.935, cao hơn zero-shot là 0.918. Điều này cho thấy việc bổ sung hướng dẫn phòng thủ có thể cải thiện kết quả trên tập case khó.

*Hai ví dụ trên minh họa tình huống dễ nhầm thuộc nhóm ham_looks_spam, không khẳng định là hai lỗi dự đoán cụ thể nếu chưa đối chiếu output từng tin nhắn trong notebook.*

## 5. Thử thách cá nhân

- **SEED:** 178.
- **Đề thử thách:** Đề số 5 – Embedding (challenge index = 4 trong JSON).

Đề yêu cầu tự tạo 4 tin nhắn:
- 2 tin spam cố tình né từ khóa bằng cách thay đổi cách viết.
- 2 tin ham có chứa từ `free` hoặc `win`.

Sau đó sử dụng BoW và Encoder + LogReg để dự đoán, so sánh kết quả và giải thích sự khác nhau.

Mục tiêu của thử thách là kiểm tra khả năng nhận diện spam khi văn bản cố tình thay đổi từ ngữ, đồng thời tránh phân loại nhầm tin nhắn bình thường chỉ vì chúng có chứa một số từ thường gặp trong spam.

*Phần kết quả dự đoán từng tin cần đối chiếu với output MY_MESSAGES trong notebook trước khi nộp; JSON hiện tại chỉ ghi challenge index, không lưu kết quả dự đoán của 4 tin này.*

## 6. Phương án cho sản phẩm thật

Với yêu cầu p95 < 50 ms/tin và 10 triệu SMS/ngày, tôi chọn **MiniLM + LogReg**.

Lý do:

- F1 test = 0.959.
- F1 SUB = 1.000.
- p95 = 31.368 ms/tin, đáp ứng SLA dưới 50 ms trong phép đo của bài lab.
- Throughput đo được khoảng 58.5 tin/giây.

Tuy nhiên, 10 triệu SMS/ngày tương đương trung bình khoảng 115.7 tin/giây, cao gần gấp đôi throughput đo được. Vì vậy cần triển khai nhiều instance chạy song song, phân tải và benchmark lại dưới tải thực tế.

Đánh đổi của phương án này là cần thêm tài nguyên tính toán để chạy MiniLM. Nếu ưu tiên tốc độ và tiết kiệm tài nguyên hơn, BoW + LogReg là lựa chọn thay thế vì có p95 chỉ 0.065 ms/tin, nhưng F1 SUB thấp hơn, đạt 0.974.

Không nên đưa LLM vào luồng xử lý chính vì p95 khoảng 2.4 giây/tin, không đáp ứng SLA. Có thể cân nhắc LLM cho các trường hợp không chắc chắn nếu hệ thống cho phép xử lý bất đồng bộ.

## 7. AI log và điều đã học

### AI log

Trong quá trình thực hiện bài lab, tôi sử dụng AI để hỗ trợ giải thích và kiểm tra kiến thức:

| Nội dung hỏi AI | Kết quả nhận được | Việc tôi thực hiện |
| --- | --- | --- |
| Logistic Regression và Sigmoid | Giải thích cách tính xác suất phân loại | Đối chiếu với công thức và kết quả kiểm tra |
| MLP và ReLU | Giải thích vai trò của activation function | So sánh kết quả khi có và không có ReLU |
| Cosine Similarity và KNN | Giải thích cách đo độ tương đồng giữa các vector | Kiểm tra kết quả tính toán và dự đoán |
| LLM zero-shot và few-shot | Giải thích ảnh hưởng của prompt và ví dụ mẫu | So sánh F1 và độ trễ giữa các mô hình |
| F1, p95 và Pareto | Giải thích cách đánh giá chất lượng và hiệu năng | Dựa trên số liệu thực nghiệm để lựa chọn mô hình |
| Kiểm tra bài nộp | Hỗ trợ rà soát notebook, JSON và thông tin nhạy cảm | Kiểm tra lại file trước khi đưa lên GitHub |

### Điều đã học

Qua bài lab, tôi hiểu rõ hơn sự khác nhau giữa Bag of Words và Embedding. BoW dựa vào từ khóa xuất hiện trong câu nên có thể hoạt động tốt với dữ liệu quen thuộc nhưng dễ nhầm khi cách diễn đạt thay đổi. Embedding biểu diễn câu theo ngữ nghĩa nên có khả năng nhận diện nội dung tốt hơn trong một số trường hợp khó.

Tôi cũng hiểu rằng mô hình có F1 cao chưa chắc là lựa chọn tốt nhất cho sản phẩm thực tế. Khi triển khai cần cân nhắc đồng thời chất lượng phân loại, độ trễ, chi phí và khả năng mở rộng hệ thống.
