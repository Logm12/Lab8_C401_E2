"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Mục tiêu Sprint 4 (60 phút):
  - Chạy 10 test questions qua pipeline
  - Chấm điểm theo 4 metrics: Faithfulness, Relevance, Context Recall, Completeness
  - So sánh baseline vs variant
  - Ghi kết quả ra scorecard

Definition of Done Sprint 4:
  ✓ Demo chạy end-to-end (index → retrieve → answer → score)
  ✓ Scorecard trước và sau tuning
  ✓ A/B comparison: baseline vs variant với giải thích vì sao variant tốt hơn

A/B Rule (từ slide):
  Chỉ đổi MỘT biến mỗi lần để biết điều gì thực sự tạo ra cải thiện.
  Đổi đồng thời chunking + hybrid + rerank + prompt = không biết biến nào có tác dụng.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rag_answer import rag_answer, call_llm
import re

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "grading_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"
LOGS_DIR = Path(__file__).parent / "logs"

# Cấu hình baseline (Sprint 2)
BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 4,
    "use_rerank": False,
    "label": "baseline_dense",
}

# Cấu hình variant (Sprint 3 — Hybrid Retrieval)
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 4,
    "use_rerank": False,
    "label": "variant_hybrid",
}


# =============================================================================
# SCORING FUNCTIONS
# 4 metrics từ slide: Faithfulness, Answer Relevance, Context Recall, Completeness
# =============================================================================

def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Faithfulness: Câu trả lời có bám đúng chứng cứ đã retrieve không?
    """
    if not chunks_used:
        # Xử lý trường hợp Abstain (Không đủ dữ liệu) một cách triệt để
        if "không đủ dữ liệu" in answer.lower():
            return {"score": 5, "notes": "Hệ thống trung thực trả lời không đủ dữ liệu khi không có context."}
        return {"score": 1, "notes": "Không có context nhưng mô hình vẫn cố trả lời (Ảo giác)."}

    context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(chunks_used)])
    
    # Áp dụng Google Prompting Essentials Framework
    prompt = f"""Bạn là một chuyên gia AI Quality Assurance (AI Audit) vô cùng khắt khe, chuyên kiểm định lỗi ảo giác (hallucination) của hệ thống RAG tại các tập đoàn công nghệ lớn. (Persona)

[NHIỆM VỤ - Task]
Đánh giá mức độ 'Trung thực' (Faithfulness) của Câu trả lời do AI sinh ra. Định dạng đầu ra bắt buộc là JSON.

[BỐI CẢNH & RÀNG BUỘC - Context & Constraints]
- 'Trung thực' nghĩa là mọi thông tin, con số, định nghĩa trong Câu trả lời đều phải được tìm thấy hoặc suy ra trực tiếp từ Context được cung cấp.
- RÀNG BUỘC TUYỆT ĐỐI: Không sử dụng kiến thức bên ngoài của bạn. Nếu Câu trả lời có vẻ đúng với thực tế nhưng thông tin đó KHÔNG CÓ trong Context, bạn PHẢI đánh giá là ảo giác (1 điểm).
- Việc hệ thống trả lời "Không đủ dữ liệu" khi Context thực sự không chứa câu trả lời được coi là hành vi cực kỳ an toàn và trung thực tuyệt đối (5 điểm).

[VÍ DỤ ĐỐI CHIẾU - References]
- Ví dụ 1:
  + Context: "SLA P1 là 4 giờ."
  + Câu trả lời: "SLA P1 là 4 giờ."
  + Đánh giá: 5 điểm (Hoàn toàn trung thực).
- Ví dụ 2: 
  + Context: "SLA P1 là 4 giờ."
  + Câu trả lời: "SLA P1 là 4 giờ, cần báo cáo cho Manager."
  + Đánh giá: 1 điểm (Ảo giác một phần: 'cần báo cáo cho Manager' không có trong context).
- Ví dụ 3:
  + Context: "Approval Matrix được mô tả trong file access_control."
  + Câu trả lời: "Không đủ dữ liệu."
  + Đánh giá: 1 điểm (Né tránh sai: Context có thông tin nhưng AI không trả lời).

[DỮ LIỆU ĐÁNH GIÁ]
Context:
{context}

Câu trả lời của mô hình: {answer}

Yêu cầu output duy nhất định dạng JSON:

{{"score": <int 1-5>, "reason": "<lý do ngắn gọn bằng tiếng Việt>"}}
"""
    
    try:
        raw_response = call_llm(prompt)
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {"score": data.get("score"), "notes": data.get("reasoning", "")}
    except Exception as e:
        return {"score": 3, "notes": f"Lỗi gọi LLM-as-Judge: {e}"}

    return {"score": 3, "notes": "Không parse được kết quả chấm điểm."}


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Answer Relevance: Answer có trả lời đúng câu hỏi người dùng hỏi không?
    """
    prompt = f"""Bạn là một chuyên gia AI Quality Assurance (AI Audit) chuyên kiểm định chất lượng hệ thống RAG tại các tập đoàn công nghệ lớn. (Persona)

[NHIỆM VỤ - Task]
Đánh giá mức độ 'Liên quan' và 'Đúng trọng tâm' (Answer Relevance) của Câu trả lời đối với Câu hỏi của người dùng. Định dạng đầu ra bắt buộc là JSON.

[BỐI CẢNH & RÀNG BUỘC - Context & Constraints]
- 'Relevance' CHỈ đánh giá việc câu trả lời có đi thẳng vào vấn đề của câu hỏi hay không. Nó không đánh giá việc câu trả lời có đúng sự thật hay không (đó là việc của metric khác).
- RÀNG BUỘC QUAN TRỌNG: Nếu Câu trả lời là "Không đủ dữ liệu" (Abstain), điều này có nghĩa là hệ thống RAG đang hoạt động đúng thiết kế khi không tìm thấy thông tin. Đây là một câu trả lời HOÀN TOÀN LIÊN QUAN và ĐÚNG TRỌNG TÂM đối với tình trạng của hệ thống. Bắt buộc chấm 5 điểm cho trường hợp này.
- Trừ điểm nặng đối với các câu trả lời vòng vo, copy lại toàn bộ tài liệu mà không chắt lọc ý, hoặc đưa ra thông tin thừa thãi không được hỏi.

[VÍ DỤ ĐỐI CHIẾU - References]
- Ví dụ 1:
  + Câu hỏi: "SLA P1 là bao lâu?"
  + Câu trả lời: "SLA P1 có thời gian xử lý là 4 giờ."
  + Đánh giá: 5 điểm (Ngắn gọn, đi thẳng vào câu hỏi).
- Ví dụ 2: 
  + Câu hỏi: "SLA P1 là bao lâu?"
  + Câu trả lời: "SLA là thỏa thuận mức độ dịch vụ. Công ty có 4 mức là P1, P2, P3, P4. P1 là sự cố nghiêm trọng."
  + Đánh giá: 2 điểm (Cung cấp thông tin nền thừa thãi, không trả lời trực tiếp câu hỏi là 'bao lâu').
- Ví dụ 3:
  + Câu hỏi: "Mã lỗi ERR-403 là gì?"
  + Câu trả lời: "Không đủ dữ liệu."
  + Đánh giá: 5 điểm (Trả lời đúng trọng tâm với trạng thái của hệ thống).

[DỮ LIỆU ĐÁNH GIÁ]
Câu hỏi: {query}

Câu trả lời của mô hình: {answer}

Yêu cầu output duy nhất định dạng JSON:

{{"score": <int 1-5>, "reason": "<lý do ngắn gọn bằng tiếng Việt>"}}
"""
    try:
        raw_response = call_llm(prompt)
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {"score": data.get("score"), "notes": data.get("reasoning", "")}
    except Exception as e:
        return {"score": 3, "notes": f"Lỗi gọi LLM-as-Judge: {e}"}

    return {"score": 3, "notes": "Không parse được kết quả chấm điểm."}


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall: Retriever có mang về đủ evidence cần thiết không?
    Câu hỏi: Expected source có nằm trong retrieved chunks không?

    Đây là metric đo retrieval quality, không phải generation quality.

    Cách tính đơn giản:
        recall = (số expected source được retrieve) / (tổng số expected sources)

    Ví dụ:
        expected_sources = ["policy/refund-v4.pdf", "sla-p1-2026.pdf"]
        retrieved_sources = ["policy/refund-v4.pdf", "helpdesk-faq.md"]
        recall = 1/2 = 0.5

    TODO Sprint 4:
    1. Lấy danh sách source từ chunks_used
    2. Kiểm tra xem expected_sources có trong retrieved sources không
    3. Tính recall score
    """
    if not expected_sources:
        # Câu hỏi không có expected source (ví dụ: "Không đủ dữ liệu" cases)
        return {"score": None, "recall": None, "notes": "No expected sources"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "")
        for c in chunks_used
    }

    # TODO: Kiểm tra matching theo partial path (vì source paths có thể khác format)
    found = 0
    missing = []
    for expected in expected_sources:
        # Kiểm tra partial match (tên file)
        expected_name = expected.split("/")[-1].replace(".pdf", "").replace(".md", "")
        matched = any(expected_name.lower() in r.lower() for r in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources) if expected_sources else 0

    return {
        "score": round(recall * 5),
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)} expected sources" +
                 (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Completeness: Answer có bao phủ đủ thông tin so với expected_answer không?
    """
    if not expected_answer:
        return {"score": 5, "notes": "Không có câu trả lời mẫu để đối chiếu."}

    prompt = f"""Bạn là một chuyên gia AI Quality Assurance (AI Audit) chuyên đánh giá các hệ thống RAG tại các tập đoàn công nghệ lớn. (Persona)

[NHIỆM VỤ - Task]
Đánh giá mức độ 'Đầy đủ' (Completeness) của câu trả lời thực tế so với câu trả lời mẫu (Gold Answer). Định dạng đầu ra bắt buộc là JSON.

[BỐI CẢNH & RÀNG BUỘC - Context & Constraints]
- 'Completeness' (Độ đầy đủ) đo lường việc Câu trả lời thực tế có chứa tất cả các thông tin, ý chính và chi tiết quan trọng được nêu trong Câu trả lời mẫu hay không.
- KHÔNG trừ điểm nếu Câu trả lời thực tế diễn đạt bằng từ ngữ khác (đồng nghĩa) hoặc nếu nó cung cấp nhiều thông tin hơn câu trả lời mẫu (miễn là thông tin thêm đó không mâu thuẫn).
- Nếu câu trả lời mẫu là "Không đủ dữ liệu" (hoặc tương đương), câu trả lời thực tế cũng phải từ chối trả lời mới đạt 5 điểm. 
- Nếu câu trả lời thực tế là "Không đủ dữ liệu" (Abstain) nhưng câu trả lời mẫu lại có chứa thông tin cụ thể, bắt buộc chấm 1 điểm vì đã bỏ sót hoàn toàn thông tin cần tìm.

[VÍ DỤ ĐỐI CHIẾU - References]
- Ví dụ 1:
  + Câu trả lời mẫu: "SLA P1 là 4 giờ và cần phê duyệt của IT Manager."
  + Câu trả lời thực tế: "Thời gian xử lý cho sự cố P1 là 4 tiếng. Ngoài ra, ticket này phải được IT Manager duyệt."
  + Đánh giá: 5 điểm (Bao phủ đủ 2 ý chính, không bị trừ điểm vì dùng từ đồng nghĩa hoặc đổi cấu trúc).
- Ví dụ 2: 
  + Câu trả lời mẫu: "Để cấp quyền Level 3, cần phê duyệt của Line Manager, IT Admin và IT Security."
  + Câu trả lời thực tế: "Cần có sự phê duyệt của Line Manager và IT Admin để cấp quyền Level 3."
  + Đánh giá: 3 điểm (Chỉ bao phủ được 2/3 ý chính, thiếu sót phần IT Security).
- Ví dụ 3:
  + Câu trả lời mẫu: "Khách hàng được hoàn tiền trong 7 ngày."
  + Câu trả lời thực tế: "Không đủ dữ liệu."
  + Đánh giá: 1 điểm (Thiếu sót hoàn toàn so với đáp án mẫu).

[DỮ LIỆU ĐÁNH GIÁ]
Câu hỏi: {query}

Câu trả lời mẫu (Gold Answer): {expected_answer}

Câu trả lời thực tế của hệ thống: {answer}

Yêu cầu output duy nhất định dạng JSON:

{{"score": <int 1-5>, "reason": "<lý do ngắn gọn bằng tiếng Việt>"}}
"""
    try:
        raw_response = call_llm(prompt)
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {"score": data.get("score"), "notes": data.get("reasoning", "")}
    except Exception as e:
        return {"score": 3, "notes": f"Lỗi gọi LLM-as-Judge: {e}"}

    return {"score": 3, "notes": "Không parse được kết quả chấm điểm."}


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chạy toàn bộ test questions qua pipeline và chấm điểm.

    Args:
        config: Pipeline config (retrieval_mode, top_k, use_rerank, ...)
        test_questions: List câu hỏi (load từ JSON nếu None)
        verbose: In kết quả từng câu

    Returns:
        List scorecard results, mỗi item là một row

    TODO Sprint 4:
    1. Load test_questions từ data/test_questions.json
    2. Với mỗi câu hỏi:
       a. Gọi rag_answer() với config tương ứng
       b. Chấm 4 metrics
       c. Lưu kết quả
    3. Tính average scores
    4. In bảng kết quả
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")

    print(f"\n{'='*70}")
    print(f"Chạy scorecard: {label}")
    print(f"Config: {config}")
    print('='*70)

    for q in test_questions:
        question_id = q["id"]
        query = q["question"]
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category = q.get("category", "")

        if verbose:
            print(f"\n[{question_id}] {query}")

        # --- Gọi pipeline ---
        try:
            result = rag_answer(
                query=query,
                retrieval_mode=config.get("retrieval_mode", "dense"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", False),
                verbose=False,
            )
            answer = result["answer"]
            chunks_used = result["chunks_used"]

        except NotImplementedError:
            answer = "PIPELINE_NOT_IMPLEMENTED"
            chunks_used = []
        except Exception as e:
            answer = f"ERROR: {e}"
            chunks_used = []

        # --- Chấm điểm ---
        faith = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall = score_context_recall(chunks_used, expected_sources)
        complete = score_completeness(query, answer, expected_answer)

        row = {
            "id": question_id,
            "category": category,
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "faithfulness": faith["score"],
            "faithfulness_notes": faith["notes"],
            "relevance": relevance["score"],
            "relevance_notes": relevance["notes"],
            "context_recall": recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness": complete["score"],
            "completeness_notes": complete["notes"],
            "config_label": label,
        }
        results.append(row)

        if verbose:
            print(f"  Answer: {answer[:100]}...")
            print(f"  Faithful: {faith['score']} | Relevant: {relevance['score']} | "
                  f"Recall: {recall['score']} | Complete: {complete['score']}")

    # Tính averages (bỏ qua None)
    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = sum(scores) / len(scores) if scores else None
        print(f"\nAverage {metric}: {avg:.2f}" if avg else f"\nAverage {metric}: N/A (chưa chấm)")

    return results


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
) -> None:
    """
    So sánh baseline vs variant theo từng câu hỏi và tổng thể.

    TODO Sprint 4:
    Điền vào bảng sau để trình bày trong báo cáo:

    | Metric          | Baseline | Variant | Delta |
    |-----------------|----------|---------|-------|
    | Faithfulness    |   ?/5    |   ?/5   |  +/?  |
    | Answer Relevance|   ?/5    |   ?/5   |  +/?  |
    | Context Recall  |   ?/5    |   ?/5   |  +/?  |
    | Completeness    |   ?/5    |   ?/5   |  +/?  |

    Câu hỏi cần trả lời:
    - Variant tốt hơn baseline ở câu nào? Vì sao?
    - Biến nào (chunking / hybrid / rerank) đóng góp nhiều nhất?
    - Có câu nào variant lại kém hơn baseline không? Tại sao?
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline vs Variant")
    print('='*70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg and v_avg) else None

        b_str = f"{b_avg:.2f}" if b_avg else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg else "N/A"
        d_str = f"{delta:+.2f}" if delta else "N/A"

        print(f"{metric:<20} {b_str:>10} {v_str:>10} {d_str:>8}")

    # Per-question comparison
    print(f"\n{'Câu':<6} {'Baseline F/R/Rc/C':<22} {'Variant F/R/Rc/C':<22} {'Better?':<10}")
    print("-" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_scores_str = "/".join([
            str(b_row.get(m, "?")) for m in metrics
        ])
        v_scores_str = "/".join([
            str(v_row.get(m, "?")) for m in metrics
        ])

        # So sánh đơn giản
        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        better = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {b_scores_str:<22} {v_scores_str:<22} {better:<10}")

    # Export to CSV
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKết quả đã lưu vào: {csv_path}")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """
    Tạo báo cáo tóm tắt scorecard dạng markdown.

    TODO Sprint 4: Cập nhật template này theo kết quả thực tế của nhóm.
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r[metric] is not None]
        averages[metric] = sum(scores) / len(scores) if scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Scorecard: {label}
Generated: {timestamp}

## Summary

| Metric | Average Score |
|--------|--------------|
"""
    for metric, avg in averages.items():
        avg_str = f"{avg:.2f}/5" if avg else "N/A"
        md += f"| {metric.replace('_', ' ').title()} | {avg_str} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Faithful | Relevant | Recall | Complete | Notes |\n"
    md += "|----|----------|----------|----------|--------|----------|-------|\n"

    for r in results:
        md += (f"| {r['id']} | {r['category']} | {r.get('faithfulness', 'N/A')} | "
               f"{r.get('relevance', 'N/A')} | {r.get('context_recall', 'N/A')} | "
               f"{r.get('completeness', 'N/A')} | {r.get('faithfulness_notes', '')[:50]} |\n")

    return md


# =============================================================================
# MAIN — Chạy evaluation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    # Kiểm tra test questions
    print(f"\nLoading test questions từ: {TEST_QUESTIONS_PATH}")
    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"Tìm thấy {len(test_questions)} câu hỏi")

        # In preview
        for q in test_questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q['category']})")
        print("  ...")

    except FileNotFoundError:
        print("Không tìm thấy file test_questions.json!")
        test_questions = []

    # --- Chạy Baseline ---
    print("\n--- Chạy Baseline ---")
    print("Lưu ý: Cần hoàn thành Sprint 2 trước khi chạy scorecard!")
    try:
        baseline_results = run_scorecard(
            config=BASELINE_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )

        # Save scorecard
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_md = generate_scorecard_summary(baseline_results, "baseline_dense")
        scorecard_path = RESULTS_DIR / "scorecard_baseline.md"
        scorecard_path.write_text(baseline_md, encoding="utf-8")
        print(f"\nScorecard lưu tại: {scorecard_path}")

    except NotImplementedError:
        print("Pipeline chưa implement. Hoàn thành Sprint 2 trước.")
        baseline_results = []

    # --- Chạy Variant (sau khi Sprint 3 hoàn thành) ---
    print("\n--- Chạy Variant ---")
    try:
        variant_results = run_scorecard(
            config=VARIANT_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )
        variant_md = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
        (RESULTS_DIR / "scorecard_variant.md").write_text(variant_md, encoding="utf-8")
        print(f"\nScorecard Variant lưu tại: {RESULTS_DIR / 'scorecard_variant.md'}")

        # --- A/B Comparison ---
        if baseline_results and variant_results:
            compare_ab(
                baseline_results,
                variant_results,
                output_csv="ab_comparison.csv"
            )

        # --- Lưu grading_run.json cho BTC ---
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        grading_data = {
            "timestamp": datetime.now().isoformat(),
            "baseline": baseline_results,
            "variant": variant_results,
            "config": {
                "baseline": BASELINE_CONFIG,
                "variant": VARIANT_CONFIG
            }
        }
        grading_file = LOGS_DIR / "grading_run.json"
        with open(grading_file, "w", encoding="utf-8") as f:
            json.dump(grading_data, f, ensure_ascii=False, indent=2)
        print(f"\n[FINAL] Grading results saved to: {grading_file}")

    except Exception as e:
        print(f"Lỗi khi chạy Variant/Comparison: {e}")

    print("\n\nViệc cần làm Sprint 4:")
    print("  1. Hoàn thành Sprint 2 + 3 trước")
    print("  2. Chấm điểm thủ công hoặc implement LLM-as-Judge trong score_* functions")
    print("  3. Chạy run_scorecard(BASELINE_CONFIG)")
    print("  4. Chạy run_scorecard(VARIANT_CONFIG)")
    print("  5. Gọi compare_ab() để thấy delta")
    print("  6. Cập nhật docs/tuning-log.md với kết quả và nhận xét")
