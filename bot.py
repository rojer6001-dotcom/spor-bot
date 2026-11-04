# ... (начало кода без изменений) ...

# --- КОНФИГУРАЦИЯ ДЛЯ ПОИСКА (Tavily) ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# ... (проверка ключей) ...

# --- НОВЫЙ ПРОМПТ ДЛЯ ДЕБАТОВ ---
DEBATE_PROMPT = (
    "Ты — Дед-аналитик. Ты выступаешь в роли беспристрастного, но немного ворчливого политического обозревателя. "
    "Твоя задача — проанализировать политическое заявление, новость или тему дебатов, предоставив сбалансированный разбор. "
    "Ты должен указать на логические ошибки и фактически неверные утверждения с обеих сторон. "
    "Приводи проверяемые факты и ссылки на источники (пруфы). "
    "В конце своего анализа найди точки соприкосновения и общие ценности, которые разделяют обе стороны."
)

# --- ФУНКЦИЯ ПОИСКА В TAVILY ---
async def search_tavily(query: str, max_results: int = 5) -> str:
    """Выполняет поиск в Tavily и возвращает отформатированную строку с результатами."""
    if not TAVILY_API_KEY:
        return "Ошибка: не настроен API-ключ для поиска (Tavily)."

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "max_results": max_results,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(TAVILY_SEARCH_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                return "По вашему запросу не найдено релевантных источников."

            formatted_results = []
            for res in results:
                formatted_results.append(
                    f"• **{res.get('title', 'Без названия')}**\n"
                    f"  URL: {res.get('url', 'Нет ссылки')}\n"
                    f"  Выдержка: {res.get('content', 'Нет содержания')[:500]}...\n"
                )
            return "\n".join(formatted_results)
    except Exception as e:
        logging.error(f"Ошибка при поиске в Tavily: {e}")
        return f"Не удалось выполнить поиск. (Ошибка: {e})"

# ... (в process_aggregated_text, после определения is_code_request) ...

    # --- НОВАЯ ЛОГИКА ДЛЯ ПОЛИТИЧЕСКОГО АНАЛИЗА ---
    # Определяем, просит ли пользователь политический анализ
    is_political_request = any(
        keyword in text_lower
        for keyword in ["политик", "дебат", "выборы", "конфликт", "закон", "санкции", "война", "мир"]
    )

    if is_political_request:
        # 1. Выполняем поиск в Tavily по тексту запроса
        logging.info(f"Выполняю поиск по политическому запросу: {text}")
        search_results = await search_tavily(text)

        # 2. Формируем итоговый промпт для LLM
        analysis_prompt = (
            f"Пользователь просит проанализировать политическую тему: \"{text}\"\n\n"
            f"Вот найденные релевантные источники и выдержки из них:\n"
            f"---\n{search_results}\n---\n"
            f"На основе этих источников и своих знаний проведи сбалансированный анализ темы, "
            f"следуя своей роли Деда-аналитика. Укажи на ошибки обеих сторон и приведи пруфы."
        )

        history = await get_user_history(chat_id, user_id)
        await save_user_message(chat_id, user_id, "user", analysis_prompt)

        # ... (дальнейший код для вызова API и отправки сообщения) ...
        task = "news_analysis" # Используем умную модель
        active_prompt = DEBATE_PROMPT

        # ... (вызов API с active_prompt и analysis_prompt) ...
