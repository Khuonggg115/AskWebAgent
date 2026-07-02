"""
streamlit_app.py - Giao diện Web UI cho Ask-the-Web Agent.

Đây là Phase 4 (Phase cuối): Tạo giao diện web bằng Streamlit.

Các tính năng chính:
1. Ô nhập câu hỏi + nút "Gửi"
2. Hiển thị tiến trình real-time bằng st.status() — nhà tuyển dụng
   có thể thấy các Node LangGraph đang nhảy
3. Kết quả markdown có trích dẫn [1], [2]
4. Nguồn tham khảo dạng link click được
5. Lịch sử chat ở Sidebar

Cách chạy:
    streamlit run streamlit_app.py
"""

import os
import streamlit as st

# ═══════════════════════════════════════════════════════
#  BƯỚC 0: Cấu hình secrets cho Streamlit Cloud
# ═══════════════════════════════════════════════════════
# Khi deploy lên Streamlit Cloud, API keys được lưu trong st.secrets
# (thay vì file .env). Đoạn code dưới đây chuyển secrets thành
# biến môi trường TRƯỚC KHI import config.py, để config.py
# hoạt động bình thường cả ở local (.env) lẫn Cloud (st.secrets).
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "TAVILY_API_KEY" in st.secrets:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except Exception:
    # Nếu rơi vào đây nghĩa là đang chạy local và chưa tạo file secrets.toml, 
    # cứ để mặc định cho python-dotenv ở file config.py tự xử lý.
    pass
# Import SAU KHI đã set env vars
from graph.build_graph import build_graph

# ═══════════════════════════════════════════════════════
#  CẤU HÌNH TRANG STREAMLIT
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="Ask-the-Web Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS tùy chỉnh ──
# Thêm một chút styling cho đẹp hơn mặc định
st.markdown("""
<style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    /* Source link styling */
    .source-item {
        padding: 8px 12px;
        margin: 4px 0;
        border-left: 3px solid #667eea;
        background-color: rgba(102, 126, 234, 0.05);
        border-radius: 0 4px 4px 0;
    }
    .source-item a {
        text-decoration: none;
        color: #667eea;
    }
    .source-item a:hover {
        text-decoration: underline;
    }
    /* Metric card */
    .metric-row {
        display: flex;
        gap: 12px;
        margin: 10px 0 20px 0;
    }
    .metric-card {
        flex: 1;
        padding: 12px 16px;
        border-radius: 8px;
        background: rgba(102, 126, 234, 0.08);
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #888;
    }
    .metric-card .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  KHỞI TẠO SESSION STATE
# ═══════════════════════════════════════════════════════
# st.session_state lưu trữ dữ liệu giữa các lần rerun của Streamlit
# (mỗi khi user tương tác, Streamlit rerun toàn bộ script từ đầu)

# Lịch sử chat: danh sách các dict {question, route, final_answer, sources, iterations}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ═══════════════════════════════════════════════════════
#  CACHE GRAPH (tránh build lại mỗi lần rerun)
# ═══════════════════════════════════════════════════════

@st.cache_resource
def get_agent():
    """
    Build graph MỘT LẦN DUY NHẤT và cache lại.
    st.cache_resource đảm bảo graph không bị rebuild
    mỗi khi user tương tác (tiết kiệm thời gian + memory).
    """
    return build_graph()


# ═══════════════════════════════════════════════════════
#  HÀM XỬ LÝ CHÍNH: Chạy agent + hiển thị real-time
# ═══════════════════════════════════════════════════════

# Mapping tên node → mô tả hiển thị trên UI
# Giúp nhà tuyển dụng thấy rõ các bước của LangGraph
NODE_DISPLAY = {
    "router": ("🔍", "Đang phân loại câu hỏi..."),
    "direct_answer": ("💬", "Đang trả lời trực tiếp..."),
    "web_search": ("🌐", "Đang tìm kiếm trên web..."),
    "synthesize_answer": ("📝", "Đang tổng hợp câu trả lời..."),
    "evaluate": ("🧠", "Đang đánh giá chất lượng câu trả lời..."),
    "need_clarification": ("❓", "Câu hỏi cần được làm rõ..."),
    "error": ("🚨", "Phát hiện lỗi hệ thống..."),
}


def run_agent_with_status(question: str) -> dict:
    """
    Chạy agent và hiển thị tiến trình real-time bằng st.status().

    Sử dụng graph.stream() thay vì graph.invoke() để nhận
    output từng node một (streaming). Mỗi khi một node hoàn thành,
    cập nhật st.status() để user thấy agent đang ở bước nào.

    Args:
        question: Câu hỏi của user

    Returns:
        dict: State cuối cùng sau khi graph chạy xong
    """
    agent = get_agent()

    # State ban đầu — giống app.py
    initial_state = {
        "question": question,
        "route": "",
        "reasoning": "",
        "search_results": [],
        "final_answer": "",
        "sources": [],
        "error_message": "",
        "needs_more_search": False,
        "search_iteration": 0,
        "next_query": "",
    }

    # Biến tích lũy state cuối cùng từ stream events
    # Mỗi node trả về dict chứa các field cần update
    # Ta merge lần lượt vào final_state
    final_state = initial_state.copy()

    # Đếm vòng search để hiển thị "Vòng X/3" trên UI
    search_count = 0

    # ── st.status(): Hiển thị tiến trình real-time ──
    # expanded=True: mở sẵn để user thấy quá trình chạy
    # Nhà tuyển dụng xem được các Node LangGraph nhảy từng bước
    with st.status("🤖 Agent đang xử lý câu hỏi...", expanded=True) as status:

        # ── graph.stream(): Nhận output từng node một ──
        # Mỗi event là dict: {"node_name": {output_fields...}}
        # Ví dụ: {"router": {"route": "web_search", "reasoning": "..."}}
        for event in agent.stream(initial_state):
            for node_name, node_output in event.items():

                # Skip node nội bộ của LangGraph (ví dụ: __start__)
                if node_name.startswith("__"):
                    continue

                # Merge output vào state tích lũy
                final_state.update(node_output)

                # Lấy icon + mô tả cho node hiện tại
                icon, description = NODE_DISPLAY.get(
                    node_name, ("⚙️", f"Đang xử lý node: {node_name}")
                )

                # ── Hiển thị chi tiết theo từng node ──

                if node_name == "router":
                    route = node_output.get("route", "")
                    reasoning = node_output.get("reasoning", "")
                    route_labels = {
                        "web_search": "🌐 Tìm kiếm Web",
                        "direct_answer": "💬 Trả lời trực tiếp",
                        "need_clarification": "❓ Cần làm rõ",
                        "error": "🚨 Lỗi hệ thống",
                    }
                    route_label = route_labels.get(route, route)
                    st.write(f"{icon} Phân loại → **{route_label}**")
                    st.caption(f"Lý do: {reasoning}")

                elif node_name == "web_search":
                    search_count += 1
                    results = node_output.get("search_results", [])
                    if search_count > 1:
                        st.write(f"🔄 **Vòng lặp Agent — Lần tìm kiếm thứ {search_count}/3**")
                        next_q = final_state.get("next_query", "")
                        if next_q:
                            st.caption(f"Query mới: \"{next_q}\"")
                    st.write(f"{icon} Tìm thấy **{len(results)}** kết quả từ web")

                elif node_name == "synthesize_answer":
                    st.write(f"{icon} Đã tổng hợp câu trả lời từ nguồn web")

                elif node_name == "evaluate":
                    iteration = node_output.get("search_iteration", 0)
                    needs_more = node_output.get("needs_more_search", False)
                    if needs_more:
                        refined = node_output.get("next_query", "")
                        st.write(f"{icon} Đánh giá: **Cần tìm thêm thông tin**")
                        st.caption(f"Query tiếp: \"{refined}\"")
                    else:
                        st.write(f"{icon} Đánh giá: **Đã đủ thông tin** ✅")

                elif node_name == "direct_answer":
                    st.write(f"{icon} Đã tạo câu trả lời trực tiếp")

                elif node_name == "need_clarification":
                    st.write(f"{icon} Câu hỏi quá mơ hồ, cần làm rõ")

                elif node_name == "error":
                    st.write(f"{icon} Phát hiện lỗi hệ thống")

        # ── Cập nhật status cuối cùng ──
        route = final_state.get("route", "")
        if route == "error":
            status.update(label="⚠️ Đã xảy ra lỗi", state="error", expanded=False)
        else:
            iterations_text = f" ({search_count} lần search)" if search_count > 0 else ""
            status.update(
                label=f"✅ Hoàn thành!{iterations_text}",
                state="complete",
                expanded=False,
            )

    return final_state


def parse_source(source_str: str) -> tuple:
    """
    Parse chuỗi source từ state["sources"] thành (label, url).

    Input format: "[1] Title — https://example.com"
    Output: ("Title", "https://example.com")

    Nếu không parse được → trả về (source_str, None)
    """
    try:
        # Tách phần " — " để lấy title và URL
        if " — " in source_str:
            label_part, url = source_str.rsplit(" — ", 1)
            # Bỏ phần [1] ở đầu label
            label = label_part.split("] ", 1)[-1] if "] " in label_part else label_part
            return (label.strip(), url.strip())
    except Exception:
        pass
    return (source_str, None)


# ═══════════════════════════════════════════════════════
#  SIDEBAR: LỊCH SỬ CHAT
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📜 Lịch sử trò chuyện")

    if not st.session_state.chat_history:
        st.caption("Chưa có câu hỏi nào. Hãy bắt đầu hỏi!")
    else:
        # Hiển thị lịch sử từ mới → cũ
        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            idx = len(st.session_state.chat_history) - i
            route_icon = {
                "web_search": "🌐",
                "direct_answer": "💬",
                "need_clarification": "❓",
                "error": "🚨",
            }.get(entry.get("route", ""), "⚙️")

            iterations = entry.get("search_iteration", 0)
            iter_badge = f" · 🔄×{iterations}" if iterations > 1 else ""

            # Mỗi câu hỏi cũ hiển thị trong expander, click để mở lại
            with st.expander(
                f"{route_icon} #{idx}: {entry['question'][:50]}{'...' if len(entry['question']) > 50 else ''}{iter_badge}",
                expanded=False,
            ):
                st.markdown(entry.get("final_answer", ""))
                sources = entry.get("sources", [])
                if sources:
                    st.caption(f"📚 {len(sources)} nguồn tham khảo")

    st.divider()
    # Nút xóa lịch sử
    if st.session_state.chat_history:
        if st.button("🗑️ Xóa lịch sử", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.caption(
        "**Ask-the-Web Agent** · Powered by LangGraph + Gemini 2.5 Flash + Tavily"
    )


# ═══════════════════════════════════════════════════════
#  MAIN AREA: GIAO DIỆN CHÍNH
# ═══════════════════════════════════════════════════════

# ── Header ──
st.markdown('<p class="main-header">🤖 Ask-the-Web Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI Agent tự động tìm kiếm web và tổng hợp câu trả lời — '
    'Powered by LangGraph + Gemini 2.5 Flash</p>',
    unsafe_allow_html=True,
)

# ── Form nhập câu hỏi ──
# Dùng st.form để tránh rerun khi user đang gõ
with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "Nhập câu hỏi của bạn:",
        placeholder="Ví dụ: So sánh React và Vue.js năm 2025...",
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submitted = st.form_submit_button(
            "🚀 Gửi câu hỏi",
            use_container_width=True,
            type="primary",
        )

# ── Gợi ý câu hỏi mẫu ──
if not st.session_state.chat_history and not submitted:
    st.markdown("#### 💡 Thử hỏi:")
    sample_cols = st.columns(3)
    samples = [
        "Giá Bitcoin hôm nay là bao nhiêu?",
        "Thủ đô của Pháp là gì?",
        "Top 5 AI framework phổ biến nhất 2025",
    ]
    for col, sample in zip(sample_cols, samples):
        with col:
            st.markdown(
                f'<div style="padding:12px;border-radius:8px;'
                f'background:rgba(102,126,234,0.08);text-align:center;'
                f'font-size:0.9rem;color:#667eea;cursor:default;">'
                f"{sample}</div>",
                unsafe_allow_html=True,
            )

# ── Xử lý khi user submit ──
if submitted and question.strip():
    question = question.strip()

    # Chạy agent + hiển thị tiến trình real-time
    result = run_agent_with_status(question)

    # ── Hiển thị metrics ──
    route = result.get("route", "")
    iterations = result.get("search_iteration", 0)
    num_sources = len(result.get("sources", []))

    route_labels = {
        "web_search": "🌐 Web Search",
        "direct_answer": "💬 Direct Answer",
        "need_clarification": "❓ Clarification",
        "error": "🚨 Error",
    }

    st.markdown(
        f"""<div class="metric-row">
            <div class="metric-card">
                <div class="label">Route</div>
                <div class="value">{route_labels.get(route, route)}</div>
            </div>
            <div class="metric-card">
                <div class="label">Vòng Search</div>
                <div class="value">{iterations}</div>
            </div>
            <div class="metric-card">
                <div class="label">Nguồn tham khảo</div>
                <div class="value">{num_sources}</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Hiển thị câu trả lời ──
    st.markdown("### 📝 Câu trả lời")
    st.markdown(result.get("final_answer", "Không có câu trả lời."))

    # ── Hiển thị nguồn tham khảo (nếu có) ──
    sources = result.get("sources", [])
    if sources:
        st.markdown("### 📚 Nguồn tham khảo")
        for source_str in sources:
            label, url = parse_source(source_str)
            if url:
                st.markdown(
                    f'<div class="source-item">'
                    f'<a href="{url}" target="_blank">🔗 {label}</a>'
                    f"<br/><small style='color:#aaa;'>{url}</small></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="source-item">{source_str}</div>',
                    unsafe_allow_html=True,
                )

    # ── Hiển thị lỗi hệ thống (nếu có) ──
    if route == "error":
        st.error(f"⚠️ {result.get('error_message', 'Lỗi không xác định')}")

    # ── Lưu vào lịch sử chat ──
    st.session_state.chat_history.append(
        {
            "question": question,
            "route": route,
            "final_answer": result.get("final_answer", ""),
            "sources": sources,
            "search_iteration": iterations,
        }
    )

    # Rerun để sidebar cập nhật lịch sử mới
    st.rerun()

# ── Hiển thị câu trả lời gần nhất từ lịch sử (sau rerun) ──
elif st.session_state.chat_history:
    latest = st.session_state.chat_history[-1]

    route = latest.get("route", "")
    iterations = latest.get("search_iteration", 0)
    sources = latest.get("sources", [])
    num_sources = len(sources)

    route_labels = {
        "web_search": "🌐 Web Search",
        "direct_answer": "💬 Direct Answer",
        "need_clarification": "❓ Clarification",
        "error": "🚨 Error",
    }

    st.markdown(
        f"""<div class="metric-row">
            <div class="metric-card">
                <div class="label">Route</div>
                <div class="value">{route_labels.get(route, route)}</div>
            </div>
            <div class="metric-card">
                <div class="label">Vòng Search</div>
                <div class="value">{iterations}</div>
            </div>
            <div class="metric-card">
                <div class="label">Nguồn tham khảo</div>
                <div class="value">{num_sources}</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### 📝 Câu trả lời")
    st.markdown(latest.get("final_answer", ""))

    if sources:
        st.markdown("### 📚 Nguồn tham khảo")
        for source_str in sources:
            label, url = parse_source(source_str)
            if url:
                st.markdown(
                    f'<div class="source-item">'
                    f'<a href="{url}" target="_blank">🔗 {label}</a>'
                    f"<br/><small style='color:#aaa;'>{url}</small></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="source-item">{source_str}</div>',
                    unsafe_allow_html=True,
                )

    if route == "error":
        st.error(f"⚠️ {latest.get('error_message', 'Lỗi không xác định')}")
