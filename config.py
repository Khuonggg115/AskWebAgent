"""
config.py - Quản lý cấu hình và API keys cho project.

Tại sao cần file này?
- Tập trung tất cả cấu hình vào MỘT chỗ, dễ bảo trì
- Load API key từ .env file (KHÔNG hardcode trong code)
- Nếu thiếu API key → báo lỗi rõ ràng ngay từ đầu
"""

import os
from dotenv import load_dotenv

# ── Load biến môi trường từ file .env ──
# load_dotenv() sẽ đọc file .env ở thư mục gốc project
# và đưa các biến vào os.environ
load_dotenv()

# ── Google Gemini API Key ──
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Kiểm tra ngay khi import: nếu thiếu key thì raise lỗi sớm
# Điều này giúp bạn biết lỗi ngay lúc khởi động, không phải đợi
# đến khi gọi API mới thấy lỗi
if not GOOGLE_API_KEY:
    raise ValueError(
        "❌ Chưa cấu hình GOOGLE_API_KEY!\n"
        "👉 Bước 1: Copy file .env.example thành .env\n"
        "👉 Bước 2: Điền API key thật vào file .env\n"
        "👉 Lấy key tại: https://aistudio.google.com/apikey"
    )

# ── Tavily Search API Key (Phase 2) ──
# Dùng cho web search khi router chọn route="web_search"
# Không raise lỗi ngay vì agent vẫn hoạt động được với direct_answer
# và need_clarification mà không cần Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    print(
        "⚠️  Chưa cấu hình TAVILY_API_KEY — tính năng Web Search sẽ không hoạt động.\n"
        "👉 Đăng ký miễn phí tại: https://app.tavily.com/sign-in\n"
        "👉 Sau đó thêm TAVILY_API_KEY=... vào file .env"
    )

# ── Cấu hình Model ──
# Dùng Gemini Flash: nhanh, rẻ, phù hợp cho routing task
GEMINI_MODEL = "gemini-3.1-flash-lite"

