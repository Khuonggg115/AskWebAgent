"""
router.py - Node phân loại câu hỏi (routing).

Đây là "bộ não" quyết định câu hỏi của user thuộc loại nào:
1. web_search:          Cần tìm thông tin mới trên internet
2. direct_answer:       LLM tự trả lời được (kiến thức phổ thông)
3. need_clarification:  Câu hỏi mơ hồ, cần hỏi lại user

Cơ chế Error Handling (cập nhật cho Phase 3):
- Lỗi HỆ THỐNG (rate limit, API fail, JSON parse) → route = "error"
  Tránh vòng lặp vô hạn khi bị rate limit ở Phase 3
- Lỗi NGỮ NGHĨA (câu hỏi mơ hồ) → route = "need_clarification"
  Đây là quyết định CÓ CHỦ ĐÍCH của router, không phải lỗi

Cách hoạt động:
- Gửi câu hỏi + prompt hướng dẫn đến Gemini API
- Gemini trả về JSON: {"route": "...", "reasoning": "..."}
- Parse JSON → ghi vào state
"""

import json

from google import genai
from google.genai.types import GenerateContentConfig

# ── Import exceptions cụ thể từ Google API ──
# google.api_core.exceptions chứa các lỗi HTTP chuẩn từ Google API
# ResourceExhausted = HTTP 429 (rate limit / quota hết)
# Đặt alias ngắn gọn để code dễ đọc hơn
from google.api_core import exceptions as gemini_exceptions

from graph.state import AgentState
import config

# ── Khởi tạo Gemini client ──
# Client này sẽ được dùng lại cho mọi lần gọi (không cần tạo mới mỗi lần)
client = genai.Client(api_key=config.GOOGLE_API_KEY)

# ── Prompt Template cho Router ──
# Prompt này hướng dẫn LLM cách phân loại câu hỏi
# Lưu ý: prompt engineering rõ ràng = kết quả routing chính xác hơn
ROUTER_PROMPT = """\
Bạn là một bộ phân loại câu hỏi (question router). Nhiệm vụ của bạn là
phân tích câu hỏi của người dùng và quyết định cách xử lý phù hợp nhất.

## Các loại route:

1. **web_search**: Chọn khi câu hỏi cần thông tin MỚI, THỜI SỰ, hoặc
   dữ liệu THAY ĐỔI theo thời gian.
   Ví dụ: "giá bitcoin hôm nay", "tin tức AI mới nhất",
   "thời tiết Hà Nội", "tỷ giá USD/VND"

2. **direct_answer**: Chọn khi câu hỏi thuộc kiến thức PHỔ THÔNG,
   KHÔNG thay đổi theo thời gian, và bạn TỰ TIN trả lời đúng.
   Ví dụ: "thủ đô Pháp là gì", "công thức tính diện tích hình tròn",
   "Python là gì", "ai viết Romeo và Juliet"

3. **need_clarification**: Chọn khi câu hỏi quá MƠ HỒ, THIẾU NGỮ CẢNH,
   hoặc có THỂ HIỂU NHIỀU CÁCH khác nhau.
   Ví dụ: "cái đó là gì", "làm sao để tốt hơn", "so sánh cho tôi"

## Quy tắc:
- Nếu KHÔNG CHẮC CHẮN giữa web_search và direct_answer → chọn web_search
  (vì tìm kiếm luôn cho kết quả cập nhật hơn)
- Trả lời CHÍNH XÁC bằng JSON format, KHÔNG thêm text nào khác

## Câu hỏi của người dùng:
{question}

## Trả lời (JSON):
{{"route": "<web_search|direct_answer|need_clarification>", "reasoning": "<giải thích ngắn gọn lý do>"}}
"""


def router_node(state: AgentState) -> dict:
    """
    Node phân loại câu hỏi sử dụng Gemini API.

    Input:  state chứa "question" (câu hỏi của user)
    Output: dict với "route", "reasoning", và có thể "error_message"

    Cơ chế xử lý lỗi (tách biệt lỗi hệ thống vs lỗi ngữ nghĩa):
    - ResourceExhausted (429) → route="error" (KHÔNG retry ở đây)
      Phase 3 sẽ xử lý retry với backoff ở tầng trên
    - JSONDecodeError           → route="error" (LLM trả sai format)
    - Exception khác            → route="error" + log chi tiết để debug
    """
    question = state["question"]
    print(f"\n🔍 Router đang phân tích câu hỏi: \"{question}\"")

    try:
        # ── Bước 1: Gọi Gemini API ──
        # generate_content() gửi prompt đến Gemini và nhận response
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=ROUTER_PROMPT.format(question=question),
            config=GenerateContentConfig(
                # temperature thấp = output ổn định, ít "sáng tạo"
                # Phù hợp cho task phân loại vì cần kết quả nhất quán
                temperature=0.0,
            ),
        )

        # ── Bước 2: Lấy text từ response ──
        raw_text = response.text.strip()

        # ── Bước 3: Parse JSON ──
        # Gemini có thể trả về JSON trong code block ```json ... ```
        # Nên cần xử lý cả 2 trường hợp: có và không có code block
        if raw_text.startswith("```"):
            # Loại bỏ code block markers (```json ... ```)
            raw_text = raw_text.split("\n", 1)[1]  # bỏ dòng ```json
            raw_text = raw_text.rsplit("```", 1)[0]  # bỏ dòng ``` cuối
            raw_text = raw_text.strip()

        result = json.loads(raw_text)

        route = result.get("route", "need_clarification")
        reasoning = result.get("reasoning", "Không có lý do")

        # Validate: đảm bảo route nằm trong danh sách hợp lệ
        valid_routes = {"web_search", "direct_answer", "need_clarification"}
        if route not in valid_routes:
            print(f"⚠️ Route không hợp lệ: '{route}', dùng 'need_clarification'")
            route = "need_clarification"
            reasoning = f"Route gốc '{route}' không hợp lệ, fallback về need_clarification"

        print(f"✅ Route: {route}")
        print(f"💡 Lý do: {reasoning}")

        # ── Thành công: trả về route + reasoning, KHÔNG có error ──
        return {"route": route, "reasoning": reasoning}

    except gemini_exceptions.ResourceExhausted:
        # ── LỖI RATE LIMIT (HTTP 429) ──
        # Đây là lỗi HỆ THỐNG, KHÔNG phải lỗi của user
        # Route = "error" để Phase 3 có thể xử lý retry với backoff
        # KHÔNG fallback về "need_clarification" vì sẽ gây vòng lặp vô hạn:
        #   need_clarification → hỏi lại user → user trả lời → router → 429 → lặp lại...
        print("❌ Lỗi Rate Limit (HTTP 429): Đã vượt quá giới hạn gọi API")
        return {
            "route": "error",
            "reasoning": "Hệ thống bị rate limit từ Gemini API",
            "error_message": (
                "Hệ thống đang quá tải do có quá nhiều yêu cầu cùng lúc. "
                "Vui lòng đợi 1 phút rồi thử lại nhé ông bạn!"
            ),
        }

    except json.JSONDecodeError as e:
        # ── LỖI PARSE JSON ──
        # Gemini trả về text không đúng format JSON
        # Đây là lỗi HỆ THỐNG (LLM không tuân thủ format),
        # KHÔNG phải câu hỏi của user mơ hồ
        print(f"❌ Lỗi parse JSON từ Gemini: {e}")
        print(f"   Raw response: {raw_text}")
        return {
            "route": "error",
            "reasoning": f"LLM trả về response không đúng format JSON: {e}",
            "error_message": (
                "Có lỗi xảy ra trong quá trình xử lý phản hồi cấu trúc từ AI."
            ),
        }

    except Exception as e:
        # ── LỖI KHÔNG XÁC ĐỊNH ──
        # Bắt tất cả lỗi còn lại: mất mạng, API key sai, server Gemini down, v.v.
        # In log chi tiết ra terminal để developer debug
        print(f"❌ Lỗi không xác định khi gọi Gemini API:")
        print(f"   Loại lỗi: {type(e).__name__}")
        print(f"   Chi tiết: {e}")
        return {
            "route": "error",
            "reasoning": f"Lỗi hệ thống không xác định: {type(e).__name__}",
            "error_message": f"Đã xảy ra lỗi hệ thống: {e}",
        }
