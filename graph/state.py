"""
state.py - Định nghĩa AgentState cho LangGraph.

AgentState là "bộ nhớ" chung mà TẤT CẢ các node trong graph
đều đọc/ghi được. Mỗi node nhận state → xử lý → trả về dict
chứa các field cần cập nhật.

Tại sao dùng TypedDict?
- TypedDict giúp IDE gợi ý code (autocomplete)
- Dễ đọc: nhìn vào là biết state có những field gì
- LangGraph yêu cầu state phải là TypedDict hoặc Pydantic model
"""

from typing import TypedDict


class AgentState(TypedDict):
    """
    State chính của agent, được truyền qua tất cả các node.

    Fields:
        question:          Câu hỏi gốc mà user nhập vào
        route:             Kết quả phân loại từ router node
                           ("web_search" | "direct_answer" | "need_clarification" | "error")
        reasoning:         Lý do router chọn route đó (để debug & học)
        search_results:    Kết quả từ web search
        final_answer:      Câu trả lời cuối cùng trả về cho user
        sources:           Danh sách URL nguồn tham khảo
        error_message:     Thông báo lỗi hệ thống (rate limit, API lỗi, v.v.)
                           Chỉ có giá trị khi route = "error"
                           Tách biệt với "need_clarification" (lỗi ngữ nghĩa user)

        --- Phase 3: Multi-step Reasoning (Agentic RAG) ---
        needs_more_search: Cờ đánh dấu: True nếu evaluate_node quyết định cần
                           search thêm để bổ sung thông tin còn thiếu.
                           Mặc định False.
        search_iteration:  Bộ đếm số lần đã search. Dùng để khống chế
                           vòng lặp tối đa (MAX = 3), tránh infinite loop.
                           Mặc định 0.
        next_query:        Query mới do LLM viết lại (refined query), tập trung
                           vào phần thông tin còn thiếu. Nếu đã đủ thông tin
                           thì để trống "".
    """

    question: str
    route: str
    reasoning: str
    search_results: list
    final_answer: str
    sources: list
    error_message: str

    # Phase 3: Multi-step Reasoning
    needs_more_search: bool
    search_iteration: int
    next_query: str
