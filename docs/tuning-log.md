# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 2026-04-13
**Config:**
```
retrieval_mode = "dense"
chunk_size = ~400 chars (paragraph-based)
overlap = ~100 chars
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = "gpt-4o-mini"
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 5 /5 |
| Answer Relevance | 4 /5 |
| Context Recall | 3 /5 |
| Completeness | 4 /5 |

**Câu hỏi yếu nhất (điểm thấp):**
1. q07 (Approval Matrix) - Context recall thấp vì Dense không nhận ra keyword cũ trong doc.
2. q09 (ERR-403) - Trả về code linh tinh thay vì từ chối nhanh.

---

## Variant 1 (Sprint 3)

**Ngày:** 2026-04-13
**Biến thay đổi:** `retrieval_mode` từ `"dense"` -> `"hybrid"`
**Lý do chọn biến này:**
Chọn hybrid vì q07 (alias query) và q09 (mã lỗi) đều cần khớp từ khóa chính xác (keyword matching). 
Corpus có chứa thông tin "Tài liệu này trước đây có tên 'Approval Matrix for System Access'" - đây là gold evidence mà BM25 sẽ túm được dễ hơn Dense.

**Config thay đổi:**
```
retrieval_mode = "hybrid"
# Các tham số còn lại giữ nguyên như baseline (A/B Rule)
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 5/5 | 5/5 | 0 |
| Answer Relevance | 4/5 | 5/5 | +1 |
| Context Recall | 3/5 | 5/5 | +2 |
| Completeness | 4/5 | 4/5 | 0 |

**Nhận xét:**
- **Ưu điểm:** Hybrid giúp tìm được chính xác chunk chứa metadata cũ ("Approval Matrix"). 
- **Lưu ý:** Trong log chạy thử, mặc dù Hybrid tìm được đúng chunk but LLM vẫn có thể Abstain nếu context bị nhiễu. Tuy nhiên, về mặt Retrieval (Recall), Hybrid vượt trội hơn hẳn ở các query có mã lỗi và tên riêng.

**Kết luận:**
Variant 1 (Hybrid) tốt hơn baseline cho các hệ thống có nhiều thuật ngữ chuyên môn và mã lỗi.
Bằng chứng: Top 1 Chunk Score của Hybrid thường ổn định hơn cho các query kỹ thuật.

---

## Variant 2 (nếu có thời gian)

**Biến thay đổi:** ___________  
**Config:**
```
# TODO
```

**Scorecard Variant 2:**
| Metric | Baseline | Variant 1 | Variant 2 | Best |
|--------|----------|-----------|-----------|------|
| Faithfulness | ? | ? | ? | ? |
| Answer Relevance | ? | ? | ? | ? |
| Context Recall | ? | ? | ? | ? |
| Completeness | ? | ? | ? | ? |

---

## Tóm tắt học được

## Tóm tắt học được

**Sau khi hoàn thành Evaluation (Sprint 4):**

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Lỗi "Context Recall" ở các query sử dụng thuật ngữ không đồng nhất (alias). Dense Retrieval mặc dù mạnh về ngữ nghĩa nhưng lại dễ bỏ lỡ các từ khóa cực kỳ quan trọng nếu chúng không xuất hiện thường xuyên hoặc có embedding vector nằm ở ranh giới khác nhau (ví dụ: "SOP" và "Approval Matrix").

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > `retrieval_mode (Hybrid)`. Việc kết hợp BM25 giúp cải thiện đáng kể độ chính xác cho các câu hỏi tra cứu theo keyword chuyên ngành, mã lỗi và giúp hệ thống "Abstain" (từ chối trả lời) một cách chắc chắn hơn khi không tìm thấy keyword match.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Tôi sẽ thử thêm bước **Reranking** bằng Cross-Encoder để tinh lọc lại top-k chunks. Điều này sẽ giúp loại bỏ các "noise" từ BM25 (vốn chỉ đếm từ mà không hiểu nghĩa) trước khi đưa vào context block của LLM, từ đó tăng điểm Faithfulness.
