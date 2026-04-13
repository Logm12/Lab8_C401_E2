# Báo Cáo Cá Nhân - Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Đoàn Hiếu  
**Vai trò trong nhóm:** Sprint 4 Developer (Evaluation, Scorecard, Results)  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500-800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi phụ trách phần `eval.py` và thư mục `results`, tức là phần đánh giá chất lượng toàn bộ pipeline RAG sau khi nhóm hoàn thành indexing và retrieval. Ở `eval.py`, tôi xây dựng cấu hình để chạy hai phiên bản của hệ thống gồm `BASELINE_CONFIG` và `VARIANT_CONFIG`, sau đó viết luồng `run_scorecard()` để đưa bộ câu hỏi chấm điểm đi qua pipeline `rag_answer()`. Tôi cũng triển khai các hàm đánh giá theo bốn tiêu chí chính là `score_faithfulness`, `score_answer_relevance`, `score_context_recall` và `score_completeness`. Bên cạnh đó, tôi viết hàm `compare_ab()` để so sánh baseline và variant theo từng metric, rồi dùng `generate_scorecard_summary()` để xuất báo cáo markdown vào thư mục `results`. Kết quả cuối cùng gồm `scorecard_baseline.md`, `scorecard_variant.md` và `ab_comparison.csv`.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, tôi hiểu rõ hơn rằng một hệ thống RAG không thể chỉ được đánh giá bằng cảm giác “trả lời có vẻ đúng”. Nếu không có scorecard, rất khó biết hệ thống thực sự tốt hơn ở đâu, kém đi ở đâu, và biến nào tạo ra khác biệt. Khi làm `eval.py`, tôi hiểu sâu hơn vai trò của từng metric. `Faithfulness` đo xem câu trả lời có bám sát context hay không; `Answer Relevance` kiểm tra câu trả lời có đi đúng trọng tâm câu hỏi; `Context Recall` đánh giá phần retrieval có kéo đúng bằng chứng về hay chưa; còn `Completeness` đo mức bao phủ so với đáp án kỳ vọng. Tôi cũng hiểu rõ hơn nguyên tắc A/B testing trong RAG: chỉ nên thay một biến mỗi lần để giải thích được tại sao kết quả thay đổi.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều làm tôi ngạc nhiên nhất là việc chấm điểm cho RAG khó hơn tôi nghĩ. Một câu trả lời có thể nghe rất hợp lý, nhưng vẫn bị chấm thấp về `faithfulness` nếu có thêm một chi tiết không xuất hiện trong context. Ngược lại, có những trường hợp hệ thống trả lời “không đủ dữ liệu”, và theo đúng thiết kế của RAG thì đó lại là một hành vi đúng. Khó khăn lớn nhất của tôi là thiết kế prompt cho phần LLM-as-Judge sao cho rõ ràng, ổn định, và ép được đầu ra ở dạng JSON để dễ parse trong code. Tôi cũng phải xử lý tình huống phản hồi từ LLM không đúng định dạng mong muốn, nên trong các hàm chấm điểm đều cần có fallback để hệ thống vẫn chạy được.

---

## 4. Phân tích một phần kỹ thuật tôi trực tiếp làm (150-200 từ)

Phần tôi thấy quan trọng nhất trong đóng góp của mình là thiết kế luồng đánh giá end-to-end trong `eval.py`. Hàm `run_scorecard()` là trung tâm của phần này: nó đọc bộ câu hỏi chấm điểm, chạy từng câu qua `rag_answer()` với cấu hình baseline hoặc variant, sau đó chấm đồng thời bốn metric và gom lại thành từng dòng kết quả. Tôi cũng đặc biệt chú ý đến `score_context_recall()`, vì đây là metric tách riêng chất lượng retrieval khỏi chất lượng generation. Nhờ đó, nếu câu trả lời sai nhưng retriever đã kéo đúng tài liệu về, nhóm có thể biết vấn đề nằm ở prompt hoặc generation, không phải ở bước search. Ngoài ra, hàm `compare_ab()` giúp biến kết quả thử nghiệm thành dữ liệu dễ đọc: từ kết quả hiện có trong thư mục `results`, baseline đạt Faithfulness trung bình 3.40/5 còn variant hybrid đạt 3.80/5, trong khi Relevance giữ nguyên 4.70/5 và Context Recall duy trì 5.00/5. Điều này cho thấy hybrid retrieval đã cải thiện tính trung thực của câu trả lời, dù vẫn có trade-off ở completeness.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi muốn cải thiện phần đánh giá theo hai hướng. Thứ nhất là bổ sung log chi tiết hơn cho từng lần chấm để dễ truy vết vì sao một câu bị giảm điểm. Thứ hai là mở rộng `results` bằng cách lưu thêm bảng tổng hợp delta theo từng metric. Tôi cũng muốn thử thêm một chế độ chấm thủ công bán tự động để đối chiếu với LLM-as-Judge.
