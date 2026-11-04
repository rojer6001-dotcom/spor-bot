import os
import logging
import random
import asyncio
import base64
import codecs
import time
import httpx
from collections import defaultdict
from difflib import get_close_matches
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile
from openai import AsyncOpenAI

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Text, BigInteger, select, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BOT_TOKEN = os.getenv("BOT_TOKEN")
PLUSVIBE_API_KEY = os.getenv("PLUSVIBE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

if not BOT_TOKEN or not PLUSVIBE_API_KEY:
    raise ValueError("❌ ОШИБКА: Не заданы BOT_TOKEN или PLUSVIBE_API_KEY!")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

client = AsyncOpenAI(
    api_key=PLUSVIBE_API_KEY,
    base_url="https://plusvibeapi.ru/v1",
    timeout=120.0 
)

# --- ЛИМИТЫ ---
MAX_TEXT_LENGTH = 15000  
MAX_FILE_LENGTH = 60000  
COOLDOWN_SECONDS = 3 
FILE_TTL_SECONDS = 600 
MAX_PARALLEL_REQUESTS = 3

api_semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
user_locks = defaultdict(asyncio.Lock)
pending_files_locks = defaultdict(asyncio.Lock)
user_cooldowns = {} 

# --- БАЗА ДАННЫХ ---
Base = declarative_base()

class UserMessage(Base):
    __tablename__ = "user_messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True) 
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    role: Mapped[str] = mapped_column(String(50))
    content: Mapped[str] = mapped_column(Text)

class ChatContext(Base):
    __tablename__ = "chat_contexts"
    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    message_text: Mapped[str] = mapped_column(Text)

engine = create_async_engine("sqlite+aiosqlite:///bot_memory_v2.db", echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

message_buffers = {} 
pending_files = {} 

# --- ФУНКЦИИ БД ---
async def get_user_history(chat_id: int, user_id: int) -> list:
    async with async_session() as session:
        result = await session.execute(
            select(UserMessage)
            .where(UserMessage.chat_id == chat_id, UserMessage.user_id == user_id)
            .order_by(UserMessage.id.desc()).limit(10)
        )
        messages = result.scalars().all()
        return [{"role": m.role, "content": m.content} for m in reversed(messages)]

async def save_user_message(chat_id: int, user_id: int, role: str, content: str):
    async with async_session() as session:
        session.add(UserMessage(chat_id=chat_id, user_id=user_id, role=role, content=content))
        await session.commit()

async def get_chat_context(chat_id: int) -> list:
    async with async_session() as session:
        result = await session.execute(
            select(ChatContext).where(ChatContext.chat_id == chat_id).order_by(ChatContext.id.desc()).limit(15)
        )
        messages = result.scalars().all()
        return [m.message_text for m in reversed(messages)]

async def save_chat_message(chat_id: int, message_text: str):
    async with async_session() as session:
        session.add(ChatContext(chat_id=chat_id, message_text=message_text))
        await session.commit()

async def cleanup_db():
    while True:
        await asyncio.sleep(3600 * 24) 
        try:
            async with async_session() as session:
                await session.execute(text("""
                    DELETE FROM chat_contexts WHERE id NOT IN (
                        SELECT id FROM (
                            SELECT id, ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY id DESC) as rn
                            FROM chat_contexts
                        ) WHERE rn <= 50
                    )
                """))
                await session.commit()
        except Exception as e:
            logging.error(f"Ошибка очистки БД: {e}")

# --- ФУНКЦИЯ ПОИСКА В TAVILY ---
async def search_tavily(query: str, max_results: int = 5) -> str:
    if not TAVILY_API_KEY:
        logging.warning("TAVILY_API_KEY не найден!")
        return "Внимание: API-ключ для поиска (Tavily) не настроен. Анализ будет основан только на внутренних знаниях модели."

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
        return "Не удалось выполнить поиск в интернете."

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
async def keep_typing(chat_id: int):
    try:
        while True:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
            await asyncio.sleep(4) 
    except asyncio.CancelledError:
        pass

async def send_long_message(message: types.Message, text: str):
    if not text or not text.strip(): return
    max_length = 4000
    if len(text) <= max_length:
        await message.reply(text)
        return
    for i in range(0, len(text), max_length):
        chunk = text[i:i+max_length]
        await message.reply(chunk)
        await asyncio.sleep(0.5)

def truncate_history(history: list, max_chars: int = 2000) -> list:
    truncated = []
    for msg in history:
        content = msg["content"]
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... [ТЕКСТ ОБРЕЗАН] ..."
        truncated.append({"role": msg["role"], "content": content})
    return truncated

def decode_text(data: bytes) -> str:
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1251", errors="replace")

def safe_get_content(response) -> str:
    if not response or not response.choices or not response.choices[0].message.content:
        raise RuntimeError("API вернул пустой ответ")
    return response.choices[0].message.content

# --- ПРОМПТЫ И ТРИГГЕРЫ ---
MODELS = {
    "simple_chat": "deepseek-v4.1-flash",
    "smart_chat": "gpt-6.1-sol",
    "news_analysis": "gpt-6.1-sol",
    "code_review": "gpt-6.1-sol", 
    "background": "claude-haiku-5.5",
    "image": "z-image-turbo"
}

def get_model(task_type: str) -> str:
    return MODELS.get(task_type, "claude-haiku-5.5")

AUTO_CHATTER_CHANCE = 0.05
AUTO_CHATTER_INTERVAL = 300

SYSTEM_PROMPT = "Ты — Дед-пердед..."
CODE_PROMPT = "Ты — Дед-пердед, Senior-разработчик..."
DEBATE_PROMPT = "Ты — Дед-аналитик..."

TRIGGERS = ["дед", "пердед", "дедуля", "деда", "пирдед", "дид", "дедок"]
DRAW_TRIGGERS = ["нарисуй", "сгенерируй", "изобрази"] 
CODE_TRIGGERS = ["ошибка", "баг", "traceback", "exception", "почему не работает", "исправь", "ревью", "код", "кодить", "программ", "def ", "import ", "```", "if ", "for ", "print("]
POLITICAL_TRIGGERS = ["политик", "дебат", "выборы", "конфликт", "закон", "санкции", "война", "мир", "анализ ситуации"]

# --- ЛОГИКА ОБРАБОТКИ ---
async def delayed_process(key: tuple):
    logging.info(f"BUFFER: Ожидание 1.5 сек для {key}...")
    await asyncio.sleep(1.5) 
    data = message_buffers.pop(key, None)
    if not data: 
        logging.info(f"BUFFER: Данные для {key} не найдены (возможно, склеились).")
        return
    
    message = data["message"]
    full_text = data["text"]
    logging.info(f"BUFFER: Запуск обработки текста: {full_text[:50]}...")
    
    try:
        await process_aggregated_text(message, full_text, data["is_reply"], data["is_forwarded"])
    except Exception:
        logging.exception("КРИТИЧЕСКАЯ ОШИБКА при обработке склеенного сообщения")
        await message.reply("Я устал, я мухожук...")

async def process_aggregated_text(message: types.Message, text: str, is_reply_to_bot: bool, is_forwarded: bool):
    if not message.from_user: return 
    user_id = message.from_user.id
    chat_id = message.chat.id
    key = (chat_id, user_id)
    
    if len(text) > MAX_TEXT_LENGTH:
        await message.reply(f"Слышь, щегол! Ты прислал {len(text)} символов. Максимум — {MAX_TEXT_LENGTH}.")
        return
    
    text_lower = text.lower()
    words = text_lower.split()
    
    await save_chat_message(chat_id, f"{message.from_user.first_name or 'Кто-то'}: {text[:500]}...")

    is_triggered = any(get_close_matches(word, TRIGGERS, n=1, cutoff=0.75) for word in words)
    is_news = is_forwarded or "http" in text_lower 
    is_code_request = any(trigger in text_lower for trigger in CODE_TRIGGERS) or "```" in text
    is_draw_request = any(trigger in text_lower for trigger in DRAW_TRIGGERS) and not is_code_request
    is_political_request = any(trigger in text_lower for trigger in POLITICAL_TRIGGERS) and not is_code_request
    
    is_group = message.chat.type in ["group", "supergroup"]

    logging.info(f"CLASSIFY: is_triggered={is_triggered}, is_political={is_political_request}, is_code={is_code_request}, is_draw={is_draw_request}")

    if is_group and not (is_triggered or is_reply_to_bot):
        logging.info("CLASSIFY: В группе без обращения, игнорирую.")
        return

    now = time.time()
    if user_id in user_cooldowns and now - user_cooldowns[user_id] < COOLDOWN_SECONDS:
        logging.info("CLASSIFY: Юзер в кулдауне, игнорирую.")
        return
    user_cooldowns[user_id] = now

    async with user_locks[key]:
        logging.info("LOCK: Блокировка получена, начинаю работу.")
        
        if is_draw_request:
            # ... логика рисования ...
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                logging.info("ACTION: Рисую картинку...")
                async with api_semaphore:
                    extract_response = await client.chat.completions.create(
                        model=get_model("simple_chat"),
                        messages=[{"role": "system", "content": "Выдели суть для рисования. Верни ТОЛЬКО текст промпта."}, {"role": "user", "content": text}],
                        temperature=0.1
                    )
                clean_prompt = safe_get_content(extract_response).strip()
                
                async with api_semaphore:
                    response = await client.images.generate(model=get_model("image"), prompt=clean_prompt, n=1, size="1024x1024")
                
                if not response.data: raise RuntimeError("Пустой ответ картинки")
                image_data = response.data[0]
                photo_to_send = None
                
                if image_data.url:
                    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as http_client:
                        img_response = await http_client.get(image_data.url)
                        if img_response.status_code == 200:
                            photo_to_send = BufferedInputFile(img_response.content, filename="image.jpg")
                elif image_data.b64_json:
                    photo_to_send = BufferedInputFile(base64.b64decode(image_data.b64_json), filename="image.jpg")

                if photo_to_send:
                    await message.reply_photo(photo=photo_to_send, caption=f"Держи, сударь. Промпт: {clean_prompt}")
                
            except Exception as e:
                logging.exception("Ошибка картинки")
                await message.reply("Я устал, я мухожук... (Ошибка генерации)")
            finally:
                typing_task.cancel()
            return 

        history = await get_user_history(chat_id, user_id)
        typing_task = asyncio.create_task(keep_typing(chat_id))
        
        try:
            if is_code_request:
                logging.info("ACTION: Анализ кода...")
                task = "code_review"
                active_prompt = CODE_PROMPT
                final_text = f"Проанализируй этот код или ошибку:\n\n{text}"
                
            elif is_political_request:
                logging.info("ACTION: Политический анализ с поиском Tavily...")
                task = "news_analysis"
                active_prompt = DEBATE_PROMPT
                
                search_results = await search_tavily(text)
                logging.info(f"TAVILY: Получены результаты: {search_results[:100]}...")
                
                final_text = (
                    f"Пользователь просит проанализировать политическую тему: \"{text}\"\n\n"
                    f"Вот найденные источники:\n---\n{search_results}\n---\n"
                    f"Проведи сбалансированный анализ темы с пруфами."
                )
            else:
                logging.info("ACTION: Обычный текстовый чат...")
                task = "smart_chat" if (is_triggered or is_reply_to_bot) else "simple_chat"
                active_prompt = SYSTEM_PROMPT
                final_text = text

            await save_user_message(chat_id, user_id, "user", final_text)
            safe_history = truncate_history(history[-4:], max_chars=2000) 
            safe_history.append({"role": "user", "content": final_text})

            logging.info(f"API: Отправка запроса к {get_model(task)}...")
            async with api_semaphore:
                response = await client.chat.completions.create(
                    model=get_model(task),
                    messages=[{"role": "system", "content": active_prompt}] + safe_history,
                    temperature=0.7 if task in ["code_review", "news_analysis"] else 0.85
                )
            
            logging.info("API: Ответ получен, отправляю в чат.")
            bot_reply = safe_get_content(response)
            await send_long_message(message, bot_reply)
            await save_user_message(chat_id, user_id, "assistant", bot_reply)

        except Exception as e:
            logging.exception("КРИТИЧЕСКАЯ ОШИБКА API")
            await message.reply("Я устал, я мухожук... (Проблемы с API)")
        finally:
            typing_task.cancel()

# --- ОБРАБОТЧИК ФАЙЛОВ ---
@dp.message(F.document)
async def handle_document(message: types.Message):
    logging.info(f"FILE: Получен документ {message.document.file_name}")
    # ... (логика файлов без изменений, только с логами) ...
    pass 

# --- ОБРАБОТЧИКИ ТЕКСТА ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Приветствую, сударь. Я умею спорить, рисовать, искать пруфы в политике и разбирать твой кривой код. Кидай текст или файлы.")

@dp.message(F.text)
async def handle_message(message: types.Message):
    if not message.text or not message.from_user: return
    
    logging.info(f"MSG: Получено сообщение от {message.from_user.id}: {message.text[:50]}")
    
    user_id = message.from_user.id
    chat_id = message.chat.id
    key = (chat_id, user_id)
    
    if message.text.startswith('/'): return
    
    is_reply_to_bot = bool(message.reply_to_message and message.reply_to_message.from_user and message.reply_to_message.from_user.id == bot.id)
    is_forwarded = message.forward_origin is not None

    if key in message_buffers:
        logging.info(f"MSG: Добавляю в буфер к существующему.")
        message_buffers[key]["text"] += "\n" + message.text
        message_buffers[key]["is_reply"] = message_buffers[key]["is_reply"] or is_reply_to_bot
        message_buffers[key]["is_forwarded"] = message_buffers[key]["is_forwarded"] or is_forwarded
        if message_buffers[key]["task"]:
            message_buffers[key]["task"].cancel()
    else:
        logging.info(f"MSG: Создаю новый буфер.")
        message_buffers[key] = {
            "text": message.text, 
            "message": message,
            "is_reply": is_reply_to_bot,
            "is_forwarded": is_forwarded
        }
        
    task = asyncio.create_task(delayed_process(key))
    message_buffers[key]["task"] = task

# ... (фоновая задача и main) ...

async def main():
    # Отключаем drop_pending_updates, чтобы не терять сообщения
    await bot.delete_webhook(drop_pending_updates=False) 
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logging.info("База данных готова.")

    bg_task = asyncio.create_task(background_chatter())
    cleanup_task = asyncio.create_task(cleanup_db())
    
    try:
        await dp.start_polling(bot)
    except Exception:
        logging.exception("Критическая ошибка polling")
    finally:
        tasks_to_stop = [bg_task, cleanup_task]
        for data in message_buffers.values():
            if data.get("task"): tasks_to_stop.append(data["task"])
            
        for t in tasks_to_stop:
            t.cancel()
            
        await asyncio.gather(*tasks_to_stop, return_exceptions=True)
        await client.close()
        await engine.dispose()
        logging.info("Все задачи остановлены.")
