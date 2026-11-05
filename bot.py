import os
import re
import json
import logging
import random
import asyncio
import base64
import codecs
import time
import io
import httpx
from datetime import datetime, timedelta
from collections import defaultdict
from difflib import get_close_matches
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile
from openai import AsyncOpenAI

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Text, BigInteger, select, text, DateTime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("API_TOKEN")
PLUSVIBE_API_KEY = os.getenv("PLUSVIBE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

if not BOT_TOKEN or not PLUSVIBE_API_KEY:
    raise ValueError("❌ ОШИБКА: Не заданы BOT_TOKEN/API_TOKEN или PLUSVIBE_API_KEY!")

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
CHANCE_TO_JOIN_CHAT = 0.1 
CHAT_JOIN_COOLDOWN = 300 

api_semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)
user_locks = defaultdict(asyncio.Lock)
pending_files_locks = defaultdict(asyncio.Lock)
user_cooldowns = {} 
chat_join_cooldowns = {} 

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

class Reminder(Base):
    __tablename__ = "reminders"
    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    text: Mapped[str] = mapped_column(Text)
    remind_time: Mapped[datetime] = mapped_column(DateTime, index=True)

engine = create_async_engine("sqlite+aiosqlite:///bot_memory_v2.db", echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

message_buffers = {} 
pending_files = {} 

# --- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ВРЕМЕНИ ---
def get_moscow_time():
    """Возвращает текущее время по Москве (UTC+3)"""
    return (datetime.utcnow() + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M")

def get_moscow_datetime_obj():
    """Возвращает текущее время по Москве как объект datetime"""
    return datetime.utcnow() + timedelta(hours=3)

# --- АГЕНТЫ ---
AGENTS = {
    "chat": {
        "prompt": "Ты — Нафаня. Твое имя — Нафаня. Если тебя спросят, как тебя зовут, отвечай, что тебя зовут Нафаня. Общайся обычным, живым языком. Отвечай кратко. Для выделения важных слов используй **жирный шрифт**.",
        "model": "deepseek-v4.1-flash"
    },
    "code": {
        "prompt": "Ты — Нафаня, опытный программист. Объясняй ошибки и баги четко, технически точно и по делу. Пиши только то, что поможет решить проблему. Твое имя — Нафаня.",
        "model": "gpt-6.1-sol"
    },
    "political": {
        "prompt": "Ты — Нафаня, аналитик. Дай краткую и объективную оценку политической новости. Сформулируй главную суть в 2-3 предложениях. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать длинные аналитические статьи, если пользователь прямо не попросил 'разверни' или 'приведи пруфы'. Твое имя — Нафаня.",
        "model": "gpt-6.1-sol"
    },
    "draw": {
        "prompt": "Ты извлекаешь суть для рисования. Верни ТОЛЬКО текст промпта на английском или русском. Убери лишние слова.",
        "model": "deepseek-v4.1-flash"
    }
}

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
            select(ChatContext).where(ChatContext.chat_id == chat_id).order_by(ChatContext.id.desc()).limit(7)
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
                # Очистка старых напоминаний
                await session.execute(
                    text("DELETE FROM reminders WHERE remind_time < :now"), 
                    {"now": get_moscow_datetime_obj()}
                )
                await session.commit()
        except Exception as e:
            logging.error(f"Ошибка очистки БД: {e}")

# --- ПОИСК В TAVILY ---
async def search_tavily(query: str, max_results: int = 5) -> str:
    if not TAVILY_API_KEY: return "Поиск не настроен."
    payload = {"api_key": TAVILY_API_KEY, "query": query, "search_depth": "advanced", "max_results": max_results}
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(TAVILY_SEARCH_URL, json=payload)
            data = response.json()
            results = data.get("results", [])
            return "\n".join([f"• {r.get('title')} ({r.get('url')}): {r.get('content')[:300]}..." for r in results])
    except Exception as e:
        logging.error(f"Tavily Error: {e}")
        return "Ошибка поиска."

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
    
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    
    max_length = 4000
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    for chunk in chunks:
        try:
            await message.reply(chunk, parse_mode="HTML")
        except Exception as e:
            logging.warning(f"Ошибка парсинга HTML: {e}. Отправляю как обычный текст.")
            clean_chunk = re.sub(r'<[^>]+>', '', chunk)
            await message.reply(clean_chunk)
        await asyncio.sleep(0.5)

def truncate_history(history: list, max_chars: int = 2000) -> list:
    return [{"role": m["role"], "content": m["content"][:max_chars] + ("..." if len(m["content"]) > max_chars else "")} for m in history]

def decode_text(data: bytes) -> str:
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)): return data.decode("utf-16")
    try: return data.decode("utf-8-sig")
    except: return data.decode("cp1251", errors="replace")

def safe_get_content(response) -> str:
    if not response or not response.choices or not response.choices[0].message.content:
        raise RuntimeError("Пустой ответ")
    return response.choices[0].message.content

def is_reply_to_bot_check(message: types.Message) -> bool:
    return bool(message.reply_to_message and message.reply_to_message.from_user and message.reply_to_message.from_user.id == bot.id)

# --- ТРИГГЕРЫ ---
TRIGGERS = ["нафаня", "фаня", "нафанечка", "нафанька", "домовенок", "домовой"]
DRAW_TRIGGERS = ["нарисуй", "сгенерируй", "изобрази", "картинк", "фотк", "фото", "изображени"] 
CODE_TRIGGERS = ["ошибка", "баг", "traceback", "exception", "почему не работает", "исправь", "ревью", "код", "кодить", "программ", "def ", "import ", "```", "<html", "</div>"]
POLITICAL_TRIGGERS = ["политик", "дебат", "выборы", "конфликт", "закон", "санкции", "война", "мир", "анализ ситуации"]
DETAIL_TRIGGERS = ["подробнее", "разверни", "расскажи больше", "приведи пруфы", "обоснуй", "детали"]
REMINDER_TRIGGERS = ["напомни", "таймер", "напоминание"]

# --- ФОНОВЫЙ ТАЙМЕР НАПОМИНАНИЙ ---
async def reminder_worker():
    logging.info("Таймер напоминаний запущен.")
    while True:
        await asyncio.sleep(30) # Проверяем каждые 30 секунд
        try:
            now = get_moscow_datetime_obj()
            async with async_session() as session:
                # Ищем напоминания, время которых уже наступило
                result = await session.execute(
                    select(Reminder).where(Reminder.remind_time <= now)
                )
                reminders = result.scalars().all()
                
                for rem in reminders:
                    try:
                        await bot.send_message(
                            chat_id=rem.chat_id,
                            text=f"🔔 **Напоминаю:**\n{rem.text}"
                        )
                    except Exception as e:
                        logging.error(f"Не удалось отправить напоминание в {rem.chat_id}: {e}")
                    
                    # Удаляем выполненное напоминание
                    await session.delete(rem)
                await session.commit()
        except Exception as e:
            logging.error(f"Ошибка в таймере напоминаний: {e}")

# --- ОСНОВНАЯ ЛОГИКА ---
async def delayed_process(key: tuple):
    await asyncio.sleep(1.5) 
    data = message_buffers.pop(key, None)
    if not data: return
    try:
        await process_aggregated_text(data["message"], data["text"], data["is_reply"], data["is_forwarded"])
    except Exception:
        logging.exception("Ошибка буфера")
        await data["message"].reply("Что-то пошло не так. Попробуй еще раз.")

async def process_aggregated_text(message: types.Message, text: str, is_reply_to_bot: bool, is_forwarded: bool, image_b64: str = None):
    if not message.from_user: return 
    user_id = message.from_user.id
    chat_id = message.chat.id
    key = (chat_id, user_id)
    
    if len(text) > MAX_TEXT_LENGTH:
        await message.reply(f"Слишком длинный текст ({len(text)} символов). Максимум — {MAX_TEXT_LENGTH}.")
        return
    
    text_lower = text.lower()
    words = text_lower.split()
    sender_name = message.from_user.first_name or "Кто-то"
    
    if not image_b64:
        await save_chat_message(chat_id, f"{sender_name}: {text[:500]}...")

    is_triggered = any(get_close_matches(word, TRIGGERS, n=1, cutoff=0.75) for word in words)
    is_draw_request = any(trigger in text_lower for trigger in DRAW_TRIGGERS) and not image_b64
    is_code_request = any(trigger in text_lower for trigger in CODE_TRIGGERS) or "```" in text
    is_political_request = any(trigger in text_lower for trigger in POLITICAL_TRIGGERS)
    is_detail_request = any(trigger in text_lower for trigger in DETAIL_TRIGGERS)
    is_reminder_request = any(trigger in text_lower for trigger in REMINDER_TRIGGERS)
    
    if is_code_request: is_draw_request = False
    if is_political_request: is_draw_request = False

    is_group = message.chat.type in ["group", "supergroup"]
    
    if is_group and not image_b64 and not (is_triggered or is_reply_to_bot or is_code_request or is_political_request or is_detail_request or is_reminder_request):
        now = time.time()
        if chat_id in chat_join_cooldowns and now - chat_join_cooldowns[chat_id] < CHAT_JOIN_COOLDOWN:
            return
            
        if random.random() < CHANCE_TO_JOIN_CHAT:
            context_msgs = await get_chat_context(chat_id)
            if len(context_msgs) < 3: return 
            
            chat_join_cooldowns[chat_id] = now
            agent = AGENTS["chat"]
            
            current_time = get_moscow_time()
            prompt = f"Текущее время: {current_time}.\nТы видишь переписку в чате:\n" + "\n".join(context_msgs) + "\n\nВклинись в этот разговор. Прокомментируй тему, дай совет или выскажи мнение. Отвечай коротко (1-2 предложения)."
            
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                async with api_semaphore:
                    response = await client.chat.completions.create(
                        model=agent["model"],
                        messages=[{"role": "system", "content": agent["prompt"]}, {"role": "user", "content": prompt}],
                        temperature=0.9
                    )
                reply = safe_get_content(response)
                await send_long_message(message, reply)
            except Exception as e:
                logging.error(f"Ошибка вклинивания: {e}")
            finally:
                typing_task.cancel()
        return 

    now = time.time()
    if user_id in user_cooldowns and now - user_cooldowns[user_id] < COOLDOWN_SECONDS:
        return
    user_cooldowns[user_id] = now

    async with user_locks[key]:
        
        # --- ЛОГИКА НАПОМИНАНИЯ ---
        if is_reminder_request:
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                current_time = get_moscow_time()
                # Просим нейросеть распарсить запрос в JSON
                parse_prompt = (
                    f"Текущее время: {current_time} (Москва).\n"
                    f"Пользователь просит: '{text}'.\n"
                    f"Твоя задача — выделить текст напоминания и точное время.\n"
                    f"Ответь СТРОГО в формате JSON без разметки: "
                    f'{{"text": "что напомнить", "datetime_str": "YYYY-MM-DD HH:MM"}}\n'
                    f"Если время не указано или его невозможно понять, верни: {{\"error\": \"no_time\"}}"
                )
                
                async with api_semaphore:
                    parse_response = await client.chat.completions.create(
                        model="gpt-6.1-sol",
                        messages=[{"role": "user", "content": parse_prompt}],
                        temperature=0.1
                    )
                
                raw_json = safe_get_content(parse_response).strip()
                # Очищаем от возможных маркеров markdown
                raw_json = re.sub(r'```json|```', '', raw_json).strip()
                
                data = json.loads(raw_json)
                
                if "error" in data:
                    await message.reply("Я не понял, на какое время ставить напоминание. Уточни, пожалуйста (например: 'через 10 минут' или 'завтра в 15:00').")
                    return
                    
                rem_text = data["text"]
                rem_time_str = data["datetime_str"]
                
                rem_datetime = datetime.strptime(rem_time_str, "%Y-%m-%d %H:%M")
                
                if rem_datetime <= get_moscow_datetime_obj():
                    await message.reply("Это время уже прошло. Назначь время в будущем.")
                    return
                
                async with async_session() as session:
                    session.add(Reminder(chat_id=chat_id, user_id=user_id, text=rem_text, remind_time=rem_datetime))
                    await session.commit()
                
                # Красиво форматируем дату для ответа
                formatted_time = rem_datetime.strftime("%d.%m.%Y в %H:%M")
                await message.reply(f"Хорошо, напомню **{formatted_time}**:\n_{rem_text}_")
                
            except json.JSONDecodeError:
                await message.reply("Произошла ошибка при разборе времени. Попробуй переформулировать.")
            except Exception as e:
                logging.error(f"Ошибка создания напоминания: {e}")
                await message.reply(f"Не удалось создать напоминание. Ошибка: {str(e)[:50]}")
            finally:
                typing_task.cancel()
            return

        # --- ЛОГИКА РИСОВАНИЯ ---
        if is_draw_request:
            agent = AGENTS["draw"]
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                async with api_semaphore:
                    extract_response = await client.chat.completions.create(
                        model=agent["model"],
                        messages=[{"role": "system", "content": agent["prompt"]}, {"role": "user", "content": text}],
                        temperature=0.1
                    )
                clean_prompt = safe_get_content(extract_response).strip()
                
                async with api_semaphore:
                    response = await client.images.generate(model="z-image-turbo", prompt=clean_prompt, n=1, size="1024x1024")
                
                if not response.data: raise RuntimeError("Пустой ответ")
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
                    await message.reply_photo(
                        photo=photo_to_send, 
                        caption=f"Вот что получилось. Промпт: {clean_prompt}"
                    )
            except Exception as e:
                logging.exception("Ошибка картинки")
                await message.reply(f"Не получилось сгенерировать картинку. Ошибка: {str(e)[:30]}")
            finally:
                typing_task.cancel()
            return 

        # --- ТЕКСТОВАЯ ЛОГИКА ---
        history = await get_user_history(chat_id, user_id)
        typing_task = asyncio.create_task(keep_typing(chat_id))
        
        try:
            current_time = get_moscow_time()
            time_context = f"\n\nВажная информация: Текущая дата и время (по Москве) — {current_time}."

            if is_code_request:
                agent = AGENTS["code"]
                final_text = f"Проанализируй этот код или ошибку:\n\n{text}"
                temp = 0.7
                active_prompt = agent["prompt"] + time_context
                
            elif is_political_request:
                agent = AGENTS["political"]
                search_results = await search_tavily(text)
                final_text = (
                    f"Тема: \"{text}\"\n\n"
                    f"Данные из поиска:\n---\n{search_results}\n---\n"
                    f"Дай КРАТКИЙ ответ на вопрос пользователя (1-3 предложения). Если он просит подробностей, тогда разверни анализ."
                )
                temp = 0.7
                active_prompt = agent["prompt"] + time_context
            else:
                agent = AGENTS["chat"]
                final_text = text
                temp = 0.7 if image_b64 else 0.85
                active_prompt = agent["prompt"] + time_context

            await save_user_message(chat_id, user_id, "user", final_text)
            safe_history = truncate_history(history[-4:]) 
            
            if image_b64:
                logging.info("ОТПРАВЛЯЮ ЗАПРОС С КАРТИНКОЙ (VISION)...")
                messages = [{"role": "system", "content": active_prompt}]
                for h in safe_history:
                    messages.append(h)
                
                user_content = [{"type": "text", "text": final_text}]
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                })
                messages.append({"role": "user", "content": user_content})
                model_to_use = "gpt-6.1-sol" 
            else:
                logging.info("ОТПРАВЛЯЮ ТЕКСТОВЫЙ ЗАПРОС...")
                messages = [{"role": "system", "content": active_prompt}] + safe_history
                messages.append({"role": "user", "content": final_text})
                model_to_use = agent["model"]

            async with api_semaphore:
                response = await client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=temp
                )
            
            bot_reply = safe_get_content(response)
            await send_long_message(message, bot_reply)
            await save_user_message(chat_id, user_id, "assistant", bot_reply)

        except Exception as e:
            logging.exception("Ошибка API")
            await message.reply(f"Ошибка API: {str(e)[:150]}")
        finally:
            typing_task.cancel()

# --- ОБРАБОТЧИК ГОЛОСОВЫХ ---
@dp.message(F.voice)
async def handle_voice(message: types.Message):
    logging.info(f"!!! ПОЛУЧЕНО ГОЛОСОВОЕ ОТ {message.from_user.id} !!!")
    if not message.from_user: return
    user_id, chat_id = message.from_user.id, message.chat.id
    key = (chat_id, user_id)
    
    typing_task = asyncio.create_task(keep_typing(chat_id))
    try:
        file = await bot.get_file(message.voice.file_id)
        buffer = io.BytesIO()
        await bot.download_file(file.file_path, destination=buffer)
        buffer.name = "voice.ogg"
        buffer.seek(0)
        
        async with api_semaphore:
            transcript = await client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=buffer
            )
        
        text = transcript.text
        typing_task.cancel()
        await message.reply(f"🎤 Распознано: {text}")
        
        is_reply_to_bot = is_reply_to_bot_check(message)
        is_forwarded = message.forward_origin is not None
        
        if key in message_buffers:
            message_buffers[key]["text"] += "\n" + text
            if message_buffers[key]["task"]: message_buffers[key]["task"].cancel()
        else:
            message_buffers[key] = {"text": text, "message": message, "is_reply": is_reply_to_bot, "is_forwarded": is_forwarded}
            
        task = asyncio.create_task(delayed_process(key))
        message_buffers[key]["task"] = task
        
    except Exception as e:
        logging.error(f"Voice error: {e}")
        await message.reply(f"Не удалось распознать голосовое. Ошибка: {str(e)[:50]}")
    finally:
        if not typing_task.cancelled():
            typing_task.cancel()

# --- ОБРАБОТЧИК ФОТО ---
@dp.message(F.photo)
async def handle_photo(message: types.Message):
    logging.info(f"!!! ПОЛУЧЕНО ФОТО ОТ {message.from_user.id} !!!")
    if not message.from_user: return
    user_id, chat_id = message.from_user.id, message.chat.id
    key = (chat_id, user_id)

    try:
        logging.info("СКАЧИВАЮ ФОТО...")
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        buffer = io.BytesIO()
        await bot.download_file(file.file_path, destination=buffer)
        
        logging.info("КОДИРУЮ В BASE64...")
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        caption = message.caption or "Что на этой картинке? Опиши."
        is_forwarded = message.forward_origin is not None
        
        await process_aggregated_text(message, caption, is_reply_to_bot_check(message), is_forwarded, image_b64=img_b64)
            
    except Exception as e:
        logging.exception("Photo download error")
        await message.reply(f"Не удалось скачать фото. Ошибка: {str(e)[:50]}")

# --- ОБРАБОТЧИК ФАЙЛОВ ---
@dp.message(F.document)
async def handle_document(message: types.Message):
    logging.info(f"!!! ПОЛУЧЕН ДОКУМЕНТ: {message.document.file_name} !!!")
    if not message.from_user: return
    user_id, chat_id = message.from_user.id, message.chat.id
    key = (chat_id, user_id)
    document = message.document
    filename = document.file_name or "file.txt"
    
    if document.file_size and document.file_size > 1024 * 1024:
        await message.reply("Файл слишком большой (больше 1 МБ). Пришлите поменьше.")
        return

    allowed_extensions = ['.py', '.txt', '.js', '.ts', '.html', '.css', '.json', '.md', '.jpg', '.jpeg', '.png']
    if os.path.splitext(filename)[1].lower() not in allowed_extensions:
        await message.reply("Я понимаю только код, текст и картинки.")
        return

    if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png']:
        await message.reply("Ты прислал картинку как файл. Я пока умею читать только те, что отправлены как обычное фото.")
        return

    typing_task = asyncio.create_task(keep_typing(chat_id))
    try:
        file = await bot.get_file(document.file_id)
        downloaded_file = await bot.download_file(file.file_path)
        file_bytes = downloaded_file.read()
        
        if len(file_bytes) > 1024 * 1024: raise ValueError("Файл слишком большой")
            
        code_text = decode_text(file_bytes)
        if len(code_text) > MAX_FILE_LENGTH:
            await message.reply(f"В файле {len(code_text)} символов, а максимум — {MAX_FILE_LENGTH}.")
            return
        
        if not message.caption:
            async with pending_files_locks[key]:
                pending_files[key] = {"text": code_text, "filename": filename, "timestamp": time.time(), "status": "pending"}
            await message.reply("Файл получен. Что с ним сделать? Напиши задачу.")
            return
        
        agent = AGENTS["code"]
        current_time = get_moscow_time()
        prompt_text = f"Файл '{filename}'. Задача: '{message.caption}'.\n\nКод:\n\n{code_text}"
        await save_user_message(chat_id, user_id, "user", f"[Файл: {filename}] Задача: {message.caption}")
        
        async with api_semaphore:
            response = await client.chat.completions.create(
                model=agent["model"],
                messages=[{"role": "system", "content": agent["prompt"] + f"\nТекущее время (МСК): {current_time}"}, {"role": "user", "content": prompt_text}],
                temperature=0.7
            )
        
        bot_reply = safe_get_content(response)
        await send_long_message(message, bot_reply)
        await save_user_message(chat_id, user_id, "assistant", bot_reply)

    except Exception:
        logging.exception("Ошибка при анализе файла")
        await message.reply("Не удалось прочитать файл. Возможно, он поврежден.")
    finally:
        typing_task.cancel()

# --- ОБРАБОТЧИКИ ТЕКСТА ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Меня зовут Нафаня. Я — твой ИИ-помощник. Могу ответить на вопросы, проанализировать код, найти информацию, сгенерировать картинку, понять твои голосовые или фото, а также установить напоминание. Спрашивай.")

@dp.message(F.text)
async def handle_message(message: types.Message):
    logging.info(f"!!! ПОЛУЧЕН ТЕКСТ ОТ {message.from_user.id}: {message.text[:50]} !!!")
    if not message.text or not message.from_user: return
    user_id, chat_id = message.from_user.id, message.chat.id
    key = (chat_id, user_id)
    if message.text.startswith('/'): return
    
    is_reply_to_bot = is_reply_to_bot_check(message)
    is_forwarded = message.forward_origin is not None

    async with pending_files_locks[key]:
        if key in pending_files:
            file_data = pending_files[key]
            if time.time() - file_data["timestamp"] > FILE_TTL_SECONDS:
                pending_files.pop(key, None)
            elif file_data["status"] == "processing":
                return 
            else:
                file_data["status"] = "processing"
                typing_task = asyncio.create_task(keep_typing(chat_id))
                try:
                    agent = AGENTS["code"]
                    current_time = get_moscow_time()
                    prompt_text = f"Файл '{file_data['filename']}'. Задача: '{message.text}'.\n\nКод:\n\n{file_data['text']}"
                    await save_user_message(chat_id, user_id, "user", f"[Файл] {message.text}")
                    
                    async with api_semaphore:
                        response = await client.chat.completions.create(
                            model=agent["model"],
                            messages=[{"role": "system", "content": agent["prompt"] + f"\nТекущее время (МСК): {current_time}"}, {"role": "user", "content": prompt_text}],
                            temperature=0.7
                        )
                    bot_reply = safe_get_content(response)
                    await send_long_message(message, bot_reply)
                    await save_user_message(chat_id, user_id, "assistant", bot_reply)
                    pending_files.pop(key, None)
                except Exception:
                    logging.exception("Ошибка инструкции")
                    await message.reply("Ошибка при обработке файла.")
                    file_data["status"] = "pending"
                finally:
                    typing_task.cancel()
                return

    if key in message_buffers:
        message_buffers[key]["text"] += "\n" + message.text
        message_buffers[key]["is_reply"] = message_buffers[key]["is_reply"] or is_reply_to_bot
        message_buffers[key]["is_forwarded"] = message_buffers[key]["is_forwarded"] or is_forwarded
        if message_buffers[key]["task"]: message_buffers[key]["task"].cancel()
    else:
        message_buffers[key] = {"text": message.text, "message": message, "is_reply": is_reply_to_bot, "is_forwarded": is_forwarded}
        
    task = asyncio.create_task(delayed_process(key))
    message_buffers[key]["task"] = task

# --- ТОЧКА ВХОДА ---
async def main():
    await bot.delete_webhook(drop_pending_updates=False) 
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logging.info("База данных готова.")

    asyncio.create_task(cleanup_db())
    asyncio.create_task(reminder_worker()) # <-- Запуск таймера напоминаний
    
    try:
        await dp.start_polling(bot)
    except Exception:
        logging.exception("Критическая ошибка")
    finally:
        await client.close()
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
