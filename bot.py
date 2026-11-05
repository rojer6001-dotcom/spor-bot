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
JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID") # <-- Ключ Jamendo
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

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def get_moscow_time():
    return (datetime.utcnow() + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M")

def get_moscow_datetime_obj():
    return datetime.utcnow() + timedelta(hours=3)

def format_text(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    return text

async def search_jamendo(query: str) -> dict | None:
    if not JAMENDO_CLIENT_ID:
        return None
    url = "https://api.jamendo.com/v3.0/tracks/"
    params = {
        "client_id": JAMENDO_CLIENT_ID,
        "format": "json",
        "limit": 1,
        "search": query,
        "audioformat": "mp32"
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            response = await http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                return None
            track = results[0]
            return {
                "name": track.get("name", "Неизвестный трек"),
                "artist": track.get("artist_name", "Неизвестный исполнитель"),
                "audio_url": track.get("audio", ""),
            }
    except Exception as e:
        logging.error(f"Ошибка поиска в Jamendo: {e}")
        return None

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

async def get_active_reminders(chat_id: int, user_id: int) -> list:
    async with async_session() as session:
        result = await session.execute(
            select(Reminder).where(Reminder.chat_id == chat_id, Reminder.user_id == user_id).order_by(Reminder.remind_time)
        )
        return result.scalars().all()

async def delete_reminders_by_query(chat_id: int, user_id: int, text_query: str = None):
    async with async_session() as session:
        if text_query:
            stmt = select(Reminder).where(
                Reminder.chat_id == chat_id, 
                Reminder.user_id == user_id,
                Reminder.text.ilike(f"%{text_query}%")
            )
            result = await session.execute(stmt)
            for rem in result.scalars().all():
                await session.delete(rem)
        else:
            stmt = select(Reminder).where(Reminder.chat_id == chat_id, Reminder.user_id == user_id)
            result = await session.execute(stmt)
            for rem in result.scalars().all():
                await session.delete(rem)
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
    
    formatted_text = format_text(text)
    max_length = 4000
    chunks = [formatted_text[i:i+max_length] for i in range(0, len(formatted_text), max_length)]
    
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
MUSIC_TRIGGERS = ["найди песню", "включи музыку", "поиск музыки", "найди трек", "поставь песню"]

REMINDER_CREATE_TRIGGERS = ["напомни", "поставь таймер", "создай напоминание", "напоминание на"]
REMINDER_LIST_TRIGGERS = ["какие таймеры", "мои напоминания", "список напоминаний", "активные таймеры", "покажи напоминания", "какие есть напоминания"]
REMINDER_DELETE_TRIGGERS = ["удали таймер", "удали напоминание", "отмени напоминание", "убери таймер", "удали все напоминания", "отмени все таймеры"]

# --- ФОНОВЫЙ ТАЙМЕР НАПОМИНАНИЙ ---
async def reminder_worker():
    logging.info("Таймер напоминаний запущен.")
    while True:
        await asyncio.sleep(30) 
        try:
            now = get_moscow_datetime_obj()
            async with async_session() as session:
                result = await session.execute(
                    select(Reminder).where(Reminder.remind_time <= now)
                )
                reminders = result.scalars().all()
                
                for rem in reminders:
                    try:
                        formatted_rem = format_text(f"🔔 **Напоминаю:**\n{rem.text}")
                        await bot.send_message(
                            chat_id=rem.chat_id,
                            text=formatted_rem,
                            parse_mode="HTML"
                        )
                    except Exception as e:
                        logging.error(f"Не удалось отправить напоминание в {rem.chat_id}: {e}")
                        await bot.send_message(chat_id=rem.chat_id, text=f"🔔 Напоминаю:\n{rem.text}")
                    
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
    
    if image_b64:
        is_draw_request = False
    else:
        is_draw_request = any(trigger in text_lower for trigger in DRAW_TRIGGERS)

    is_code_request = any(trigger in text_lower for trigger in CODE_TRIGGERS) or "```" in text
    is_political_request = any(trigger in text_lower for trigger in POLITICAL_TRIGGERS)
    is_detail_request = any(trigger in text_lower for trigger in DETAIL_TRIGGERS)
    is_music_request = any(trigger in text_lower for trigger in MUSIC_TRIGGERS)
    
    is_reminder_create = any(trigger in text_lower for trigger in REMINDER_CREATE_TRIGGERS)
    is_reminder_list = any(trigger in text_lower for trigger in REMINDER_LIST_TRIGGERS)
    is_reminder_delete = any(trigger in text_lower for trigger in REMINDER_DELETE_TRIGGERS)

    if is_reminder_delete or is_reminder_list:
        is_reminder_create = False
    
    if is_code_request: is_draw_request = False
    if is_political_request: is_draw_request = False
    if is_music_request: 
        is_draw_request = False
        is_code_request = False
        is_political_request = False

    is_group = message.chat.type in ["group", "supergroup"]
    
    if is_group and not image_b64 and not (is_triggered or is_reply_to_bot or is_code_request or is_political_request or is_detail_request or is_reminder_create or is_reminder_list or is_reminder_delete or is_music_request):
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
        
        # --- ЛОГИКА ПОИСКА МУЗЫКИ ---
        if is_music_request:
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                extract_prompt = f"Из запроса пользователя вычлени только исполнителя и название песни для поиска. Верни ТОЛЬКО поисковый запрос, без лишних слов. Запрос: '{text}'"
                async with api_semaphore:
                    extract_response = await client.chat.completions.create(
                        model="deepseek-v4.1-flash",
                        messages=[{"role": "user", "content": extract_prompt}],
                        temperature=0.1
                    )
                search_query = safe_get_content(extract_response).strip()
                
                track = await search_jamendo(search_query)
                
                if track and track.get("audio_url"):
                    await message.reply_audio(
                        audio=track["audio_url"],
                        title=track["name"],
                        performer=track["artist"],
                        caption=f"Нашел трек: **{track['name']}** — {track['artist']}"
                    )
                else:
                    await message.reply(f"Не удалось найти трек по запросу «{search_query}». Попробуй переформулировать.")
            except Exception as e:
                logging.exception("Ошибка поиска музыки")
                await message.reply("Не удалось найти музыку. Попробуй позже.")
            finally:
                typing_task.cancel()
            return

        # ... (Логика напоминаний и рисования без изменений) ...
        if is_reminder_list:
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                reminders = await get_active_reminders(chat_id, user_id)
                if not reminders:
                    await message.reply("У тебя нет активных напоминаний.")
                else:
                    reply_text = "**Твои активные напоминания:**\n\n"
                    for i, rem in enumerate(reminders, 1):
                        t_str = rem.remind_time.strftime("%d.%m.%Y в %H:%M")
                        reply_text += f"{i}. **{t_str}** — {rem.text}\n"
                    
                    formatted_text = format_text(reply_text)
                    await message.reply(formatted_text, parse_mode="HTML")
            except Exception as e:
                logging.error(f"Ошибка получения списка напоминаний: {e}")
                await message.reply("Не удалось получить список напоминаний.")
            finally:
                typing_task.cancel()
            return

        if is_reminder_delete:
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                current_time = get_moscow_time()
                parse_prompt = (
                    f"Текущее время: {current_time}.\n"
                    f"Пользователь пишет: '{text}'.\n"
                    f"Он хочет удалить напоминание(я).\n"
                    f"Твоя задача — понять, что именно он хочет удалить.\n"
                    f"Если он хочет удалить ВСЕ напоминания (например: 'удали все'), верни JSON: {{\"delete_all\": true}}.\n"
                    f"Если он хочет удалить конкретное напоминание, извлеки ключевое слово из его текста и верни JSON: {{\"query\": \"ключевое слово\"}}.\n"
                    f"Ответь СТРОГО в формате JSON без разметки."
                )
                
                async with api_semaphore:
                    parse_response = await client.chat.completions.create(
                        model="gpt-6.1-sol",
                        messages=[{"role": "user", "content": parse_prompt}],
                        temperature=0.1
                    )
                
                raw_json = safe_get_content(parse_response).strip()
                raw_json = re.sub(r'```json|```', '', raw_json).strip()
                
                data = json.loads(raw_json)
                
                if data.get("delete_all"):
                    await delete_reminders_by_query(chat_id, user_id, text_query=None)
                    await message.reply("Хорошо, я удалил **все** твои напоминания.")
                elif data.get("query"):
                    query = data["query"]
                    await delete_reminders_by_query(chat_id, user_id, text_query=query)
                    await message.reply(f"Понял. Если было напоминание со словом **«{query}»**, я его удалил.")
                else:
                    await message.reply("Я не понял, какое именно напоминание нужно удалить. Уточни.")
                    
            except json.JSONDecodeError:
                await message.reply("Произошла ошибка при разборе запроса на удаление.")
            except Exception as e:
                logging.error(f"Ошибка удаления напоминания: {e}")
                await message.reply("Не удалось удалить напоминание.")
            finally:
                typing_task.cancel()
            return

        if is_reminder_create:
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                current_time = get_moscow_time()
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
                raw_json = re.sub(r'```json|```', '', raw_json).strip()
                
                data = json.loads(raw_json)
                
                if "error" in data:
                    await message.reply("Я не понял, на какое время ставить напоминание. Уточни, пожалуйста.")
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
                
                formatted_time = rem_datetime.strftime("%d.%m.%Y в %H:%M")
                reply_text = format_text(f"Хорошо, напомню **{formatted_time}**:\n_{rem_text}_")
                await message.reply(reply_text, parse_mode="HTML")
                
            except json.JSONDecodeError:
                await message.reply("Произошла ошибка при разборе времени. Попробуй переформулировать.")
            except Exception as e:
                logging.error(f"Ошибка создания напоминания: {e}")
                await message.reply(f"Не удалось создать напоминание. Ошибка: {str(e)[:50]}")
            finally:
                typing_task.cancel()
            return

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

        # ИСПРАВЛЕННЫЙ БЛОК EXCEPT
        except Exception as e:
            error_str = str(e).lower()
            logging.exception("Ошибка API")
            if "connection" in error_str or "timeout" in error_str:
                await message.reply("Ой, кажется, у меня пропала связь с сервером. Попробуй отправить сообщение еще раз через минуту.")
            else:
                await message.reply("Произошла ошибка при обработке запроса. Попробуй позже.")
        finally:
            typing_task.cancel()

# ... (ОСТАЛЬНОЙ КОД: голосовые, фото, файлы, текст - БЕЗ ИЗМЕНЕНИЙ) ...
# (Вставьте сюда обработчики голосовых, фото, файлов и handle_message из предыдущего полного кода)
