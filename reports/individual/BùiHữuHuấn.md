# Báo Cáo Cá Nhân - Lab Day 08: RAG Pipeline

**Họ và tên:** Bùi Hữu Huấn  
**Vai trò trong nhóm:** Documentation Owner (Architecture, Tuning Log, Report)  
**Ngày nộp:** 13/04/2026  
**Độ dài:** ~650 từ  

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi đảm nhiệm vai trò Documentation Owner, chịu trách nhiệm mô tả toàn bộ kiến trúc hệ thống RAG và ghi lại quá trình tuning pipeline của nhóm. Cụ thể, tôi xây dựng file `architecture.md` để mô tả đầy đủ các thành phần chính của hệ thống gồm indexing, retrieval, generation và evaluation. Trong phần indexing, tôi tổng hợp cấu trúc dữ liệu, chiến lược chunking và metadata. Trong phần retrieval, tôi trình bày rõ baseline (dense) và variant (hybrid + RRF), cùng với lý do lựa chọn.  

Ngoài ra, tôi viết file `tuning-log.md` để ghi lại quá trình thử nghiệm theo nguyên tắc A/B testing. Tôi so sánh hiệu năng giữa baseline và hybrid retrieval, phân tích sự thay đổi của các metric như faithfulness và completeness, đồng thời chỉ ra nguyên nhân và trade-off. Hai tài liệu này đóng vai trò giúp nhóm hiểu rõ hệ thống đang hoạt động như thế nào và vì sao lại chọn hướng tối ưu hiện tại.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, tôi hiểu rõ hơn rằng việc xây dựng một hệ thống RAG không chỉ là viết code mà còn là quá trình thiết kế hệ thống có chủ đích. Trước đây, tôi thường nghĩ retrieval chỉ đơn giản là tìm kiếm vector, nhưng khi làm documentation, tôi nhận ra rằng lựa chọn chiến lược retrieval (dense vs hybrid) phụ thuộc rất nhiều vào đặc điểm dữ liệu.  

Tôi cũng hiểu rõ hơn vai trò của chunking và metadata. Nếu chunking không hợp lý, retrieval sẽ không thể lấy đúng ngữ cảnh, dẫn đến toàn bộ pipeline bị sai. Metadata không chỉ để lưu trữ mà còn phục vụ explainability và filtering. Ngoài ra, việc ghi tuning log giúp tôi hiểu rõ nguyên tắc A/B testing: chỉ thay đổi một biến mỗi lần để có thể giải thích được nguyên nhân của sự cải thiện hoặc suy giảm hiệu năng.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều làm tôi ngạc nhiên nhất là việc documentation lại ảnh hưởng trực tiếp đến chất lượng hệ thống. Khi viết lại kiến trúc và tuning log, tôi phát hiện ra một số vấn đề mà trước đó nhóm không nhận ra, ví dụ như việc hybrid retrieval làm giảm completeness do hệ thống trở nên quá “an toàn” và từ chối trả lời nhiều hơn.  

Khó khăn lớn nhất là việc chuyển các quyết định kỹ thuật thành ngôn ngữ rõ ràng, dễ hiểu nhưng vẫn chính xác. Ví dụ, khi giải thích tại sao chọn hybrid retrieval, tôi phải cân nhắc giữa việc mô tả lý thuyết (semantic vs keyword search) và đưa ra ví dụ thực tế (SLA P1, ERR-403). Ngoài ra, việc phân tích kết quả tuning cũng không đơn giản vì có những trade-off không rõ ràng ngay từ đầu.

---

## 4. Phân tích một phần kỹ thuật tôi trực tiếp làm (150-200 từ)

Phần tôi thấy quan trọng nhất trong đóng góp của mình là thiết kế và ghi lại chiến lược retrieval trong `architecture.md` và `tuning-log.md`. Cụ thể, tôi đã phân tích sự khác biệt giữa dense retrieval và hybrid retrieval sử dụng Reciprocal Rank Fusion (RRF). Dense retrieval mạnh ở việc hiểu ngữ nghĩa, nhưng lại yếu khi gặp các từ khóa cố định như “P1”, “Level 3” hoặc tên tài liệu cũ. Ngược lại, sparse retrieval (BM25) lại xử lý tốt các trường hợp này nhưng không hiểu được ngữ cảnh.  

Việc kết hợp hai phương pháp bằng RRF giúp hệ thống tận dụng được cả hai ưu điểm. Trong tuning log, tôi ghi nhận rằng hybrid retrieval giúp tăng faithfulness vì model có context chính xác hơn, từ đó giảm hallucination. Tuy nhiên, completeness lại giảm nhẹ do hệ thống ưu tiên trả lời an toàn hơn khi không chắc chắn. Đây là một trade-off hợp lý trong môi trường doanh nghiệp, nơi độ chính xác quan trọng hơn độ đầy đủ.  

Việc ghi lại các kết quả này không chỉ giúp nhóm hiểu hệ thống mà còn tạo cơ sở để tiếp tục cải tiến, ví dụ như thêm rerank hoặc query transformation trong tương lai.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi muốn mở rộng documentation theo hướng trực quan hơn, ví dụ như thêm flow diagram chi tiết cho từng bước retrieval và generation. Tôi cũng muốn bổ sung phần phân tích lỗi (error analysis) cho từng loại câu hỏi để giúp việc tuning có định hướng rõ ràng hơn. Ngoài ra, tôi sẽ thử thêm rerank bằng cross-encoder và cập nhật tuning log để đánh giá tác động của bước này lên các metric.
