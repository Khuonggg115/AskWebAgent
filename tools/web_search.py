"""
web_search.py - Tìm kiếm thông tin trên web bằng Tavily API.

Tavily là search API được thiết kế riêng cho AI agents:
- Trả về nội dung đã được trích xuất (không phải HTML thô)
- Tối ưu cho LLM: content ngắn gọn, relevance cao
- Free tier: 1000 searches/tháng — đủ cho portfolio project

Cách hoạt động:
1. Nhận query (câu hỏi của user)
2. Gọi Tavily Search API
3. Trả về list các dict: [{"title": ..., "url": ..., "content": ...}, ...]
"""

from tavily import TavilyClient

import config


def search_web(query: str, max_results: int = 5) -> list:
    """
    Tìm kiếm thông tin trên web bằng Tavily API.

    Args:
        query:       Câu hỏi / từ khóa tìm kiếm
        max_results: Số kết quả tối đa cần lấy (mặc định 5)

    Returns:
        list[dict]: Danh sách kết quả, mỗi kết quả có format:
                    {"title": str, "url": str, "content": str}
                    Trả về list rỗng nếu có lỗi hoặc không tìm thấy

    Raises:
        Không raise exception — tất cả lỗi được xử lý bên trong
        và trả về list rỗng (để node gọi hàm này không bị crash)
    """

    # ── Kiểm tra API key trước khi gọi ──
    # Nếu chưa cấu hình key → trả về rỗng ngay, không gọi API
    if not config.TAVILY_API_KEY:
        print("❌ TAVILY_API_KEY chưa được cấu hình!")
        print("   👉 Thêm TAVILY_API_KEY=... vào file .env")
        return []

    try:
        # ── Khởi tạo Tavily client ──
        # Tạo client mới mỗi lần gọi (Tavily client nhẹ, không cần cache)
        client = TavilyClient(api_key=config.TAVILY_API_KEY)

        print(f"   🔎 Đang tìm kiếm: \"{query}\"")

        # ── Gọi Tavily Search API ──
        # search() trả về dict chứa key "results" là list các kết quả
        # Mỗi kết quả có: title, url, content, score, raw_content
        response = client.search(
            query=query,
            max_results=max_results,
            # search_depth="basic": tìm nhanh, phù hợp cho hầu hết câu hỏi
            # Dùng "advanced" nếu cần kết quả chi tiết hơn (chậm hơn, tốn credit hơn)
            search_depth="basic",
        )

        # ── Chuẩn hóa kết quả ──
        # Chỉ lấy 3 field cần thiết: title, url, content
        # Bỏ qua score, raw_content, v.v. để giữ state gọn nhẹ
        results = []
        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", "Không có tiêu đề"),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                }
            )

        print(f"   ✅ Tìm thấy {len(results)} kết quả")
        return results

    except Exception as e:
        # ── Xử lý lỗi ──
        # Có thể do: API key sai, hết quota, mất mạng, Tavily server down
        # Trả về list rỗng để synthesize_answer_node xử lý fallback
        print(f"   ❌ Lỗi khi tìm kiếm web: {type(e).__name__}: {e}")
        return []
