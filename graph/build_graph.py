"""
build_graph.py - Lắp ráp StateGraph cho agent.

Đây là nơi kết nối tất cả các node thành một workflow hoàn chỉnh.

Luồng graph (Phase 1 + Phase 2 + Phase 3):

    START ──► router ──┬──► direct_answer ──────────────────────────► END
                       ├──► need_clarification ─────────────────────► END
                       ├──► error ──────────────────────────────────► END
                       │
                       └──► web_search ──► synthesize ──► evaluate ──┐
                              ▲                                      │
                              │          ┌── (đủ thông tin) ────────► END
                              │          │
                              └──────────┴── (cần thêm & iteration < 3)
                                              quay lại web_search
                                              với query mới

Phase 3 thêm:
- evaluate_node: Gemini tự đánh giá câu trả lời đã đủ chưa (Self-Correction)
- Vòng lặp có kiểm soát: web_search → synthesize → evaluate → (lặp lại hoặc END)
- Giới hạn tối đa 3 vòng lặp (MAX_SEARCH_ITERATIONS) tránh infinite loop
- next_query: LLM tự viết lại query mới tập trung vào phần thiếu
"""

import json

from langgraph.graph import StateGraph, START, END

from google import genai
from google.genai.types import GenerateContentConfig

from graph.state import AgentState
from graph.router import router_node
# Import hàm search_web (không phải node) — node sẽ được định nghĩa ở đây
from tools.web_search import search_web
import config

# ── Hằng số ──
# Giới hạn tối đa số vòng lặp search → synthesize → evaluate
# Tại sao là 3?
# - 1 lần search đầu: bao phủ phần lớn câu hỏi
# - 2-3 lần tiếp: bổ sung chi tiết thiếu cho câu hỏi phức tạp
# - Quá 3 lần: diminishing returns, tốn API credits, gây chậm
MAX_SEARCH_ITERATIONS = 3

# ── Khởi tạo Gemini client ──
# Client dùng chung cho direct_answer_node, synthesize_answer_node, evaluate_node
client = genai.Client(api_key=config.GOOGLE_API_KEY)

# ── Prompt cho Direct Answer ──
DIRECT_ANSWER_PROMPT = """\
Hãy trả lời câu hỏi sau một cách rõ ràng, chính xác và dễ hiểu.
Nếu câu hỏi bằng tiếng Việt, trả lời bằng tiếng Việt.
Nếu câu hỏi bằng tiếng Anh, trả lời bằng tiếng Anh.

Câu hỏi: {question}
"""

# ── Prompt cho Synthesize Answer (Phase 2) ──
SYNTHESIZE_PROMPT = """\
Bạn là một trợ lý AI thông minh. Nhiệm vụ của bạn là trả lời câu hỏi của
người dùng dựa trên các kết quả tìm kiếm web bên dưới.

## Quy tắc QUAN TRỌNG:
1. CHỈ sử dụng thông tin từ kết quả tìm kiếm để trả lời. KHÔNG tự bịa thêm.
2. Trích dẫn nguồn bằng cách ghi số [1], [2], [3]... tương ứng với thứ tự
   kết quả tìm kiếm bên dưới.
3. Nếu các kết quả tìm kiếm KHÔNG chứa thông tin liên quan đến câu hỏi,
   hãy trả lời CHÍNH XÁC: "Tôi đã tìm kiếm trên web nhưng không thấy
   thông tin đáng tin cậy về chủ đề này, và tôi sẽ không tự bịa ra thông tin."
4. Trả lời bằng CÙNG NGÔN NGỮ với câu hỏi (Việt → Việt, Anh → Anh).
5. Trả lời rõ ràng, mạch lạc, dễ hiểu.

## Câu hỏi:
{question}

## Kết quả tìm kiếm:
{search_context}

## Trả lời:
"""

# ── Prompt cho Evaluate Node (Phase 3) ──
# Prompt này hướng dẫn LLM tự đánh giá (Self-Correction):
# Câu trả lời hiện tại đã đủ chưa? Cần search thêm gì?
#
# QUAN TRỌNG: Quy tắc viết lại query (Refined Query Rules)
# giúp LLM KHÔNG search lại thông tin đã có, mà chỉ tập trung
# vào phần thông tin còn THIẾU (Information Gap).
EVALUATE_PROMPT = """\
Bạn là một Chuyên gia Đánh giá Hệ thống AI (Agentic RAG Evaluator).
Nhiệm vụ của bạn là kiểm tra xem câu trả lời hiện tại đã đủ thông tin
giải quyết câu hỏi gốc chưa, và viết lại câu lệnh tìm kiếm nếu còn thiếu.

## Câu hỏi gốc của người dùng:
{question}

## Câu trả lời hiện tại:
{current_answer}

## Bước 1 — Đánh giá chất lượng:
Phân tích câu trả lời trên theo các tiêu chí:
- Câu trả lời có giải quyết TRỌN VẸN câu hỏi không?
- Có khía cạnh nào quan trọng còn THIẾU không?
- Thông tin có cụ thể và chi tiết ĐỦ không?

Nếu câu trả lời đã tốt (dù không hoàn hảo), hãy đánh giá is_sufficient = true.
Chỉ đánh giá is_sufficient = false khi thực sự còn khía cạnh QUAN TRỌNG
chưa được đề cập.

## Bước 2 — Viết lại query (CHỈ khi is_sufficient = false):
Tuân thủ TUYỆT ĐỐI các quy tắc sau:

1. **KHÔNG tìm kiếm lại thông tin ĐÃ CÓ**: Nếu câu trả lời đã chứa
   một thông tin cụ thể, KHÔNG ĐƯỢC đưa thông tin đó vào query mới.
   Ví dụ: Nếu đã biết "GPT-5 ra mắt ngày 07/08/2025", KHÔNG ĐƯỢC
   tìm "GPT-5 release date" nữa.

2. **Xác định Khoảng Trống Thông Tin (Information Gap)**: Làm phép trừ
   giữa "những gì câu hỏi gốc yêu cầu" và "những gì câu trả lời đã có"
   → Phần còn lại chính là thông tin cần tìm thêm.

3. **Query TẬP TRUNG 100% vào phần thiếu**: Query mới phải ngắn gọn,
   cụ thể, mang tính tra cứu sự thật (fact-checking).
   Ví dụ tốt: "Bitcoin price on August 7 2025"
   Ví dụ xấu: "latest Bitcoin information and updates"

4. **KHÔNG thêm từ khóa chung chung**: Tránh các từ như "latest",
   "overview", "everything about". Query phải nhắm vào DỮ KIỆN CỤ THỂ.

## Trả lời (JSON, KHÔNG thêm text nào khác):
{{"is_sufficient": true/false, "refined_query": "<query mới tập trung vào phần THIẾU, để trống nếu đủ>", "reasoning": "<giải thích: liệt kê thông tin ĐÃ CÓ và thông tin CÒN THIẾU>"}}
"""



def direct_answer_node(state: AgentState) -> dict:
    """
    Node trả lời trực tiếp câu hỏi kiến thức phổ thông bằng Gemini.

    Chỉ được gọi khi router quyết định route="direct_answer",
    tức câu hỏi thuộc kiến thức phổ thông mà LLM tự trả lời được.
    """
    question = state["question"]
    print(f"\n💬 Direct Answer đang trả lời: \"{question}\"")

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=DIRECT_ANSWER_PROMPT.format(question=question),
            config=GenerateContentConfig(
                # temperature 0.7: cân bằng giữa chính xác và tự nhiên
                temperature=0.7,
            ),
        )

        answer = response.text.strip()
        print(f"✅ Đã tạo câu trả lời thành công")

        return {
            "final_answer": answer,
            # direct_answer không cần source vì dùng kiến thức có sẵn của LLM
            "sources": [],
        }

    except Exception as e:
        print(f"❌ Lỗi khi tạo câu trả lời: {e}")
        return {
            "final_answer": f"Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời: {e}",
            "sources": [],
        }


# ═══════════════════════════════════════════════════════
#  PHASE 2: Web Search + Synthesize Answer Nodes
# ═══════════════════════════════════════════════════════


def web_search_node(state: AgentState) -> dict:
    """
    Node tìm kiếm web bằng Tavily API.

    Phase 3 cập nhật: Ở vòng lặp sau (search_iteration > 0),
    node này dùng state["next_query"] (query đã được LLM viết lại)
    thay vì câu hỏi gốc state["question"]. Điều này giúp tập trung
    tìm kiếm vào phần thông tin còn THIẾU, thay vì search lại y hệt.

    Luồng xử lý:
    1. Kiểm tra: dùng next_query (nếu có) hoặc question gốc
    2. Gọi hàm search_web() từ tools/web_search.py
    3. Lưu kết quả vào state["search_results"]
    """
    # ── Phase 3: Chọn query thông minh ──
    # Vòng đầu tiên (iteration 0): dùng câu hỏi gốc
    # Vòng sau (iteration > 0): dùng next_query do evaluate_node viết lại
    next_query = state.get("next_query", "")
    question = state["question"]
    iteration = state.get("search_iteration", 0)

    # Nếu có next_query (từ vòng evaluate trước) → dùng query mới
    # Nếu không → dùng câu hỏi gốc (lần search đầu tiên)
    search_query = next_query if next_query else question

    # ── In thông tin vòng lặp nổi bật ──
    if iteration > 0:
        print(f"\n🔄 {'='*50}")
        print(f"🔄 [VÒNG LẶP AGENT] Lần tìm kiếm thứ {iteration + 1}/{MAX_SEARCH_ITERATIONS}")
        print(f"🔄 Query mới: \"{search_query}\"")
        print(f"🔄 {'='*50}")
    else:
        print(f"\n🌐 Web Search Node: Đang tìm kiếm cho \"{search_query}\"")

    # ── Gọi hàm search_web ──
    # Hàm này đã xử lý error handling bên trong
    # và trả về list rỗng nếu có lỗi
    results = search_web(query=search_query, max_results=5)

    # ── In tóm tắt kết quả thô ──
    if results:
        print(f"\n📋 Kết quả thô từ web ({len(results)} nguồn):")
        for i, r in enumerate(results, 1):
            # Cắt content ngắn lại cho dễ đọc trên terminal
            snippet = r["content"][:100] + "..." if len(r["content"]) > 100 else r["content"]
            print(f"   [{i}] {r['title']}")
            print(f"       🔗 {r['url']}")
            print(f"       📄 {snippet}")
    else:
        print("   ⚠️ Không tìm thấy kết quả nào")

    # ── Cập nhật state ──
    # Phase 3: Kết quả mới THAY THẾ kết quả cũ (không cộng dồn)
    # vì synthesize_answer_node sẽ tổng hợp lại toàn bộ từ đầu
    # Reset needs_more_search về False — evaluate_node sẽ quyết định lại
    return {
        "search_results": results,
        "needs_more_search": False,
        "next_query": "",
    }


def synthesize_answer_node(state: AgentState) -> dict:
    """
    Node tổng hợp câu trả lời từ kết quả tìm kiếm web.

    Luồng xử lý:
    1. Lấy search_results từ state
    2. Nếu rỗng → trả về fallback message (KHÔNG bịa thông tin)
    3. Nếu có kết quả → format thành context, gửi cho Gemini tổng hợp
    4. Bóc tách danh sách URL sources từ search_results
    5. Trả về final_answer + sources

    Đây là node QUAN TRỌNG NHẤT của Phase 2 — nơi biến dữ liệu thô
    từ web thành câu trả lời tự nhiên có trích dẫn nguồn.
    """
    question = state["question"]
    search_results = state.get("search_results", [])
    iteration = state.get("search_iteration", 0)

    print(f"\n📝 Synthesize Node: Đang tổng hợp câu trả lời... (lần {iteration + 1})")

    # ── Fallback: Không có kết quả tìm kiếm ──
    if not search_results:
        print("   ⚠️ Không có kết quả tìm kiếm để tổng hợp")
        return {
            "final_answer": (
                "Tôi đã tìm kiếm trên web nhưng không thấy thông tin "
                "đáng tin cậy về chủ đề này, và tôi sẽ không tự bịa ra thông tin."
            ),
            "sources": [],
        }

    # ── Bước 1: Format search results thành context cho LLM ──
    search_context_parts = []
    for i, result in enumerate(search_results, 1):
        search_context_parts.append(
            f"[{i}] Tiêu đề: {result['title']}\n"
            f"    URL: {result['url']}\n"
            f"    Nội dung: {result['content']}"
        )
    search_context = "\n\n".join(search_context_parts)

    try:
        # ── Bước 2: Gọi Gemini để tổng hợp câu trả lời ──
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=SYNTHESIZE_PROMPT.format(
                question=question,
                search_context=search_context,
            ),
            config=GenerateContentConfig(
                temperature=0.3,
            ),
        )

        answer = response.text.strip()

        # ── Bước 3: Bóc tách danh sách sources ──
        sources = [
            f"[{i+1}] {r['title']} — {r['url']}"
            for i, r in enumerate(search_results)
        ]

        print(f"   ✅ Đã tổng hợp câu trả lời thành công")

        return {
            "final_answer": answer,
            "sources": sources,
        }

    except Exception as e:
        print(f"   ❌ Lỗi khi tổng hợp câu trả lời: {e}")
        return {
            "final_answer": f"Đã tìm thấy {len(search_results)} kết quả web nhưng xảy ra lỗi khi tổng hợp: {e}",
            "sources": [],
        }


# ═══════════════════════════════════════════════════════
#  PHASE 3: Evaluate Node (Self-Correction / Agentic RAG)
# ═══════════════════════════════════════════════════════


def evaluate_node(state: AgentState) -> dict:
    """
    Node tự đánh giá chất lượng câu trả lời (Self-Correction).

    Đây là trái tim của Phase 3 (Agentic RAG):
    - LLM tự hỏi: "Câu trả lời đã đủ tốt chưa?"
    - Nếu CHƯA ĐỦ: viết lại query mới → quay lại search
    - Nếu ĐÃ ĐỦ: đi thẳng tới END

    Cơ chế chống infinite loop:
    - Tăng search_iteration mỗi lần đánh giá
    - Hàm should_continue() kiểm tra iteration < MAX_SEARCH_ITERATIONS
    - Dù LLM đánh giá "chưa đủ", nếu đã chạm giới hạn → DỪNG

    Gemini trả về JSON:
    {
        "is_sufficient": bool,
        "refined_query": str,   // query mới nếu cần search thêm
        "reasoning": str
    }
    """
    question = state["question"]
    current_answer = state.get("final_answer", "")
    iteration = state.get("search_iteration", 0)

    # ── Tăng bộ đếm vòng lặp ──
    # Tăng TRƯỚC khi đánh giá để should_continue() kiểm tra đúng
    new_iteration = iteration + 1

    print(f"\n🧠 Evaluate Node: Đang đánh giá câu trả lời (vòng {new_iteration}/{MAX_SEARCH_ITERATIONS})...")

    try:
        # ── Gọi Gemini để đánh giá ──
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=EVALUATE_PROMPT.format(
                question=question,
                current_answer=current_answer,
            ),
            config=GenerateContentConfig(
                # temperature 0.0: cần đánh giá nhất quán, không "sáng tạo"
                temperature=0.0,
            ),
        )

        raw_text = response.text.strip()

        # ── Parse JSON response ──
        # Xử lý trường hợp Gemini trả JSON trong code block ```json ... ```
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]  # bỏ dòng ```json
            raw_text = raw_text.rsplit("```", 1)[0]  # bỏ ``` cuối
            raw_text = raw_text.strip()

        result = json.loads(raw_text)

        is_sufficient = result.get("is_sufficient", True)
        refined_query = result.get("refined_query", "")
        reasoning = result.get("reasoning", "Không có lý do")

        if is_sufficient:
            # ── ĐÃ ĐỦ thông tin → kết thúc ──
            print(f"   ✅ Đánh giá: ĐÃ ĐỦ thông tin")
            print(f"   💡 Lý do: {reasoning}")
            return {
                "needs_more_search": False,
                "search_iteration": new_iteration,
                "next_query": "",
            }
        else:
            # ── CẦN THÊM thông tin → chuẩn bị search tiếp ──
            print(f"   🔄 Đánh giá: CẦN TÌM THÊM thông tin")
            print(f"   💡 Lý do: {reasoning}")
            print(f"   🔎 Query mới: \"{refined_query}\"")
            return {
                "needs_more_search": True,
                "search_iteration": new_iteration,
                "next_query": refined_query,
            }

    except json.JSONDecodeError as e:
        # Lỗi parse JSON → coi như đã đủ (an toàn, không lặp thêm)
        print(f"   ⚠️ Lỗi parse JSON từ evaluate: {e}")
        print(f"   → Mặc định: coi như đã đủ thông tin (an toàn)")
        return {
            "needs_more_search": False,
            "search_iteration": new_iteration,
            "next_query": "",
        }

    except Exception as e:
        # Lỗi API → coi như đã đủ (an toàn, không lặp thêm)
        print(f"   ❌ Lỗi khi đánh giá: {e}")
        print(f"   → Mặc định: coi như đã đủ thông tin (an toàn)")
        return {
            "needs_more_search": False,
            "search_iteration": new_iteration,
            "next_query": "",
        }


def should_continue(state: AgentState) -> str:
    """
    Hàm điều kiện quyết định sau evaluate_node: TIẾP TỤC hay DỪNG.

    Đây là cơ chế CHỐNG INFINITE LOOP quan trọng nhất:

    Quy tắc:
    1. needs_more_search == True VÀ search_iteration < MAX (3)
       → Quay lại "web_search" để search tiếp với query mới
    2. Tất cả trường hợp khác → Đi tới END
       - Đã đủ thông tin (needs_more_search == False)
       - Đã chạm giới hạn 3 lần (search_iteration >= 3)

    Tại sao dùng "end" thay vì END trực tiếp?
    - LangGraph conditional edges cần return STRING
    - String này được map đến node hoặc END qua path_map

    Returns:
        str: "web_search" (lặp lại) hoặc "end" (kết thúc)
    """
    needs_more = state.get("needs_more_search", False)
    iteration = state.get("search_iteration", 0)

    if needs_more and iteration < MAX_SEARCH_ITERATIONS:
        # ── TIẾP TỤC: Còn cần thông tin VÀ chưa hết quota vòng lặp ──
        print(f"\n🔄 should_continue → TIẾP TỤC (lần {iteration + 1}/{MAX_SEARCH_ITERATIONS})")
        return "web_search"
    else:
        # ── DỪNG: Đã đủ HOẶC đã chạm giới hạn ──
        if iteration >= MAX_SEARCH_ITERATIONS:
            print(f"\n⛔ should_continue → DỪNG (đã chạm giới hạn {MAX_SEARCH_ITERATIONS} vòng)")
        else:
            print(f"\n✅ should_continue → DỪNG (câu trả lời đã đủ)")
        return "end"


# ═══════════════════════════════════════════════════════
#  Các node xử lý khác (Phase 1)
# ═══════════════════════════════════════════════════════


def need_clarification_node(state: AgentState) -> dict:
    """
    Node xử lý khi câu hỏi quá mơ hồ, cần hỏi lại user.
    Trả về message yêu cầu user làm rõ câu hỏi.

    LƯU Ý: Đây là route cho lỗi NGỮ NGHĨA (câu hỏi mơ hồ),
    KHÔNG phải lỗi hệ thống. Lỗi hệ thống đi route "error".
    """
    question = state["question"]
    reasoning = state.get("reasoning", "")
    print(f"\n❓ Câu hỏi cần làm rõ: \"{question}\"")

    return {
        "final_answer": (
            f"🤔 Tôi chưa hiểu rõ câu hỏi của bạn: \"{question}\"\n"
            f"Lý do: {reasoning}\n"
            f"Bạn có thể diễn đạt lại cụ thể hơn không?"
        ),
        "sources": [],
    }


def error_node(state: AgentState) -> dict:
    """
    Node xử lý lỗi HỆ THỐNG (rate limit, API fail, JSON parse error).

    Nhiệm vụ đơn giản: lấy error_message từ state → gán vào final_answer
    → đi thẳng tới END. KHÔNG retry, KHÔNG redirect.
    """
    error_message = state.get("error_message", "Đã xảy ra lỗi không xác định.")
    print(f"\n🚨 Error Node: {error_message}")

    return {
        "final_answer": f"⚠️ {error_message}",
        "sources": [],
    }


# ═══════════════════════════════════════════════════════
#  Routing + Build Graph
# ═══════════════════════════════════════════════════════


def route_decision(state: AgentState) -> str:
    """
    Hàm quyết định routing từ router_node (Conditional Edge).

    Returns:
        str: Tên node tiếp theo ("direct_answer", "web_search",
             "need_clarification", hoặc "error")
    """
    route = state.get("route", "need_clarification")
    return route


def build_graph() -> StateGraph:
    """
    Xây dựng và trả về StateGraph hoàn chỉnh cho agent.

    Phase 3 cập nhật:
    - Thêm node "evaluate" (đánh giá câu trả lời)
    - Thay edge synthesize → END bằng synthesize → evaluate
    - Thêm conditional edge từ evaluate: loop lại web_search hoặc đi END
    - Luồng web search giờ là vòng lặp có kiểm soát (max 3 iterations)
    """

    # ── Bước 1: Khởi tạo graph với state schema ──
    graph = StateGraph(AgentState)

    # ── Bước 2: Thêm các node ──
    graph.add_node("router", router_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesize_answer", synthesize_answer_node)
    # Phase 3: Node đánh giá chất lượng câu trả lời (Self-Correction)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("need_clarification", need_clarification_node)
    graph.add_node("error", error_node)

    # ── Bước 3: Nối các edge ──

    # START → router
    graph.add_edge(START, "router")

    # Conditional Edge từ router (Phase 1)
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "direct_answer": "direct_answer",
            "web_search": "web_search",
            "need_clarification": "need_clarification",
            "error": "error",
        },
    )

    # Phase 2 + 3: web_search → synthesize → evaluate
    # (Phase 2 là web_search → synthesize → END)
    # (Phase 3 thêm evaluate SAU synthesize, tạo vòng lặp)
    graph.add_edge("web_search", "synthesize_answer")
    graph.add_edge("synthesize_answer", "evaluate")

    # Phase 3: Conditional Edge từ evaluate (VÒNG LẶP CÓ KIỂM SOÁT)
    # should_continue() trả về:
    # - "web_search" → loop lại node web_search với query mới
    # - "end"        → đi tới END (đã đủ hoặc hết quota vòng lặp)
    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "web_search": "web_search",  # ← vòng lặp quay lại đây
            "end": END,                   # ← kết thúc
        },
    )

    # Các node khác đi thẳng tới END (không thay đổi)
    graph.add_edge("direct_answer", END)
    graph.add_edge("need_clarification", END)
    graph.add_edge("error", END)

    # ── Bước 4: Compile graph ──
    compiled = graph.compile()
    print("✅ Graph đã được build thành công!")

    return compiled
