# 🤖 Ask-the-Web Agent

> **Perplexity thu nhỏ** — AI Agent tự động phân loại câu hỏi, tìm kiếm web, tự đánh giá và tìm thêm nếu chưa đủ, rồi tổng hợp câu trả lời có trích dẫn nguồn. Xây dựng bằng **LangGraph** + **Google Gemini 2.5 Flash** + **Tavily Search**.

🔗 **Live Demo**:https://askwebagent-azff3ebgvab6mkabqby9km.streamlit.app/

---

## 🏗️ Kiến trúc hệ thống

```
User Question
      │
      ▼
┌──────────┐
│  Router  │ ◄── Gemini phân loại câu hỏi (Structured Output)
└────┬─────┘
     │
     ├─── "direct_answer" ──► Gemini trả lời ────────────────────► UI
     ├─── "need_clarification" ──► Hỏi lại user ────────────────► UI
     ├─── "error" ──► Thông báo lỗi (Rate Limit, v.v.) ────────► UI
     │
     └─── "web_search" ──► Agentic Search Loop (tối đa 3 vòng):
                           │
                           ▼
                    ┌──────────────┐
                    │  Web Search  │ ◄── Tavily API
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  Synthesize  │ ◄── Gemini tổng hợp + trích dẫn [1][2]
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐     ┌─── Đủ ──► UI
                    │   Evaluate   │─────┤
                    └──────────────┘     └─── Thiếu + iteration < 3
                      (Self-Correction)       │
                           ▲                  │ refined_query
                           └──────────────────┘
```

## ✨ Điểm nổi bật

| Kỹ thuật | Giải thích |
|----------|-----------|
| **Agentic RAG** | Agent tự chủ quyết định khi nào cần search web, khi nào tự trả lời |
| **Self-Correction Loop** | LLM tự đánh giá câu trả lời, viết lại query mới nếu thiếu thông tin |
| **Structured Output** | Router + Evaluate trả về JSON chuẩn, parse an toàn |
| **Information Gap Analysis** | Evaluate node xác định "khoảng trống thông tin" trước khi search tiếp |
| **Guardrails chống bịa đặt** | Nếu search results không liên quan → trả lời thẳng "không tìm thấy" thay vì đoán bừa |
| **Error Isolation** | Tách biệt lỗi hệ thống (rate limit) vs lỗi ngữ nghĩa (câu hỏi mơ hồ) |
| **Controlled Loop** | Hard limit 3 vòng lặp + safe fallback tránh infinite loop |
| **Real-time UI** | `st.status()` hiển thị từng Node LangGraph đang chạy |

## 🛠️ Tech Stack

- **LangGraph** — Orchestration framework dạng graph (StateGraph, Conditional Edges)
- **Google Gemini 2.5 Flash** — LLM cho routing, trả lời, tổng hợp, đánh giá
- **Tavily Search API** — Web search tối ưu cho AI agents
- **Streamlit** — Web UI với real-time status tracking
- **python-dotenv** — Quản lý API keys an toàn

## 📁 Cấu trúc Project

```
ask_web_agent/
├── streamlit_app.py       # Web UI (Streamlit)
├── app.py                 # CLI entry point (terminal)
├── config.py              # Load API keys & constants
├── requirements.txt       # Dependencies
├── .env.example           # Template API keys
├── .gitignore
├── graph/
│   ├── state.py           # AgentState (TypedDict)
│   ├── router.py          # Router node (phân loại câu hỏi)
│   └── build_graph.py     # Tất cả nodes + StateGraph assembly
├── tools/
│   └── web_search.py      # Tavily Search API integration
└── README.md
```

## 🚀 Cài đặt & Chạy Local

### 1. Clone & tạo virtual environment

```bash
git clone https://github.com/<your-username>/ask-web-agent.git
cd ask-web-agent

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình API Keys

```bash
cp .env.example .env
```

Mở file `.env` và điền API keys:
```env
GOOGLE_API_KEY=your_google_api_key_here      # https://aistudio.google.com/apikey
TAVILY_API_KEY=tvly-your_tavily_key_here     # https://app.tavily.com/sign-in
```

### 4. Chạy

```bash
# Web UI (Streamlit):
streamlit run streamlit_app.py

# CLI (Terminal):
python app.py
```

## ☁️ Deploy lên Streamlit Community Cloud

1. Đẩy code lên GitHub repository
2. Vào [share.streamlit.io](https://share.streamlit.io/) → "New app"
3. Chọn repo → Main file: `streamlit_app.py`
4. Vào **Advanced Settings → Secrets**, thêm:
   ```toml
   GOOGLE_API_KEY = "your_key"
   TAVILY_API_KEY = "tvly-your_key"
   ```
5. Click **Deploy**!

## 📋 Roadmap

| Phase | Nội dung | Trạng thái |
|-------|----------|------------|
| **1** | Project setup + Router logic + Error handling | ✅ |
| **2** | Tích hợp Tavily Web Search + Synthesize Answer | ✅ |
| **3** | Multi-step Reasoning (Agentic RAG Loop) | ✅ |
| **4** | Streamlit Web UI + Deploy Cloud | ✅ |

