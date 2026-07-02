"""
app.py - Entry point chạy agent qua terminal.

Cách dùng:
    python app.py

Phase 3 cập nhật:
- Hiển thị số vòng lặp search agent đã thực hiện
- Hiển thị thông tin chi tiết hơn cho luồng web_search
"""

from graph.build_graph import build_graph


def main():
    """Hàm chính: chạy agent trong chế độ terminal."""

    print("=" * 60)
    print("🤖 Ask-the-Web Agent — Phase 3: Multi-step Reasoning")
    print("=" * 60)
    print("Agent giờ có thể TỰ ĐÁNH GIÁ câu trả lời và SEARCH THÊM")
    print("nếu thấy chưa đủ (tối đa 3 vòng lặp).")
    print("-" * 60)
    print("Gõ câu hỏi để test. Gõ 'quit' hoặc 'exit' để thoát.")
    print("Thử câu hỏi phức tạp: 'So sánh React và Vue.js năm 2025'")
    print("Hoặc câu đơn giản: 'Thủ đô Pháp là gì?'\n")

    # ── Build graph một lần duy nhất ──
    agent = build_graph()

    while True:
        # ── Nhận câu hỏi từ user ──
        try:
            question = input("\n📝 Câu hỏi của bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Tạm biệt!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("\n👋 Tạm biệt!")
            break

        if not question:
            print("⚠️  Vui lòng nhập câu hỏi!")
            continue

        # ── Chạy graph ──
        # Các node sẽ in tiến trình trực tiếp ra terminal trong quá trình chạy
        # (router → web_search → synthesize → evaluate → loop/end)
        print("\n" + "-" * 50)

        result = agent.invoke(
            {
                "question": question,
                "route": "",
                "reasoning": "",
                "search_results": [],
                "final_answer": "",
                "sources": [],
                "error_message": "",
                # Phase 3: khởi tạo các field mới
                "needs_more_search": False,
                "search_iteration": 0,
                "next_query": "",
            }
        )

        # ── In kết quả cuối cùng ──
        print("\n" + "=" * 60)
        print("📊 KẾT QUẢ CUỐI CÙNG:")
        print("=" * 60)
        print(f"❓ Câu hỏi:  {result['question']}")
        print(f"🔀 Route:    {result['route']}")
        print(f"💡 Lý do:    {result['reasoning']}")

        # Nếu route = "error": hiển thị thông báo lỗi hệ thống
        if result["route"] == "error":
            print(f"\n🚨 Lỗi hệ thống: {result['error_message']}")

        # Phase 3: Hiển thị thông tin vòng lặp nếu đi qua web_search
        if result["route"] == "web_search":
            iteration = result.get("search_iteration", 0)
            num_results = len(result.get("search_results", []))
            print(f"🌐 Số nguồn web: {num_results} kết quả")
            print(f"🔄 Số vòng search: {iteration} lần")

        print(f"\n📝 Trả lời:\n{result['final_answer']}")

        # Hiển thị danh sách nguồn tham khảo
        if result["sources"]:
            print(f"\n📚 Nguồn tham khảo:")
            for src in result["sources"]:
                print(f"   {src}")

        print("=" * 60)


if __name__ == "__main__":
    main()
