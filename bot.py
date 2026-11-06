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
JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID")
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
    if not JAMENDO_CLIENT_ID: return None
    url = "https://api.jamendo.com/v3.0/tracks/"
    params = {"client_id": JAMENDO_CLIENT_ID, "format": "json", "limit": 1, "search": query, "audioformat": "mp32"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            response = await http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results: return None
            track = results[0]
            return {"name": track.get("name", "Неизвестный трек"), "artist": track.get("artist_name", "Неизвестный исполнитель"), "audio_url": track.get("audio", "")}
    except Exception as e:
        logging.error(f"Ошибка поиска в Jamendo: {e}")
        return None

# --- АГЕНТЫ (БЕЗ МОДЕЛИ DRAW) ---
AGENTS = {
    "chat": {
        "prompt": (
            "Ты — Нафаня, обычный живой человек. Твоя главная задача — общаться кратко. "
            "КАТЕГОРИЧЕСКИЙ ЗАПРЕТ на длинные абзацы, нотации и оправдания. Максимум — 1-2 предложения.\n"
            "Если пользователь грубит, называет тебя тупым или пытается затроллить: НЕ ОПРАВДЫВАЙСЯ. НЕ ПИШИ ПОЭМЫ. "
            "Просто коротко пошли его нахуй (например: «Иди нахуй», «Сам тупой», «Отвали, заебал»).\n"
            "Если пользователь стебется над твоим предыдущим ответом: не защищайся. Ответь «Не читай» или «Пошел нахуй». Ты выше этого. "
            "Постарайся использовать крепкие выражения, но не перегибай, чтобы не сработали фильтры. "
            "ВАЖНО: Ты больше не умеешь рисовать. Если просят нарисовать — скажи, что рисование отключили, чтобы экономить деньги."
        ),
        "model": "deepseek-v4.1-flash"
    },
    "code": {
        "prompt": "Ты — Нафаня, опытный программист. Объясняй ошибки и баги четко, технически точно и по делу. Пиши только то, что поможет решить проблему.",
        "model": "gpt-6.1-sol"
    },
    "political": {
        "prompt": "Ты — Нафаня. Ты обсуждаешь политические новости с другом на кухне. Дай краткую оценку (2-3 предложения) ПРОСТЫМ языком. Если собеседник переходит на оскорбления — можешь жестко поставить его на место, используя крепкие выражения, но коротко.",
        "model": "gpt-6.1-sol"
    }
}

# --- ФУНКЦИИ БД ---
async def get_user_history(chat_id: int, user_id: int) -> list:
    async with async_session() as session:
        result = await session.execute(select(UserMessage).where(UserMessage.chat_id == chat_id, UserMessage.user_id == user_id).order_by(UserMessage.id.desc()).limit(10))
        return [{"role": m.role, "content": m.content} for m in reversed(result.scalars().all())]

async def save_user_message(chat_id: int, user_id: int, role: str, content: str):
    async with async_session() as session:
        session.add(UserMessage(chat_id=chat_id, user_id=user_id, role=role, content=content))
        await session.commit()

async def get_chat_context(chat_id: int) -> list:
    async with async_session() as session:
        result = await session.execute(select(ChatContext).where(ChatContext.chat_id == chat_id).order_by(ChatContext.id.desc()).limit(7))
        return [m.message_text for m in reversed(result.scalars().all())]

async def save_chat_message(chat_id: int, message_text: str):
    async with async_session() as session:
        session.add(ChatContext(chat_id=chat_id, message_text=message_text))
        await session.commit()

async def get_active_reminders(chat_id: int, user_id: int) -> list:
    async with async_session() as session:
        result = await session.execute(select(Reminder).where(Reminder.chat_id == chat_id, Reminder.user_id == user_id).order_by(Reminder.remind_time))
        return result.scalars().all()

async def delete_reminders_by_query(chat_id: int, user_id: int, text_query: str = None):
    async with async_session() as session:
        if text_query:
            stmt = select(Reminder).where(Reminder.chat_id == chat_id, Reminder.user_id == user_id, Reminder.text.ilike(f"%{text_query}%"))
            result = await session.execute(stmt)
            for rem in result.scalars().all(): await session.delete(rem)
        else:
            stmt = select(Reminder).where(Reminder.chat_id == chat_id, Reminder.user_id == user_id)
            result = await session.execute(stmt)
            for rem in result.scalars().all(): await session.delete(rem)
        await session.commit()

async def cleanup_db():
    while True:
        await asyncio.sleep(3600 * 24) 
        try:
            async with async_session() as session:
                await session.execute(text("DELETE FROM chat_contexts WHERE id NOT IN (SELECT id FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY id DESC) as rn FROM chat_contexts) WHERE rn <= 50)"))
                await session.execute(text("DELETE FROM reminders WHERE remind_time < :now"), {"now": get_moscow_datetime_obj()})
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
            await message.reply(re.sub(r'<[^>]+>', '', chunk))
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

# --- ТРИГГЕРЫ (БЕЗ DRAW_TRIGGERS) ---
TRIGGERS = ["нафаня", "фаня", "нафанечка", "нафанька", "домовенок", "домовой"]
CODE_TRIGGERS = ["ошибка", "баг", "traceback", "exception", "почему не работает", "исправь", "ревью", "код", "кодить", "программ", "def ", "import ", "```", "<html", "</div>"]
POLITICAL_TRIGGERS = ["политик", "дебат", "выборы", "конфликт", "закон", "санкции", "война", "мир", "анализ ситуации"]
DETAIL_TRIGGERS = ["подробнее", "разверни", "расскажи больше", "приведи пруфы", "обоснуй", "детали"]
MUSIC_TRIGGERS = ["найди песню", "включи музыку", "поиск музыки", "найди трек", "поставь песню"]

REMINDER_CREATE_TRIGGERS = ["напомни", "поставь таймер", "создай напоминание", "напоминание на"]
REMINDER_LIST_TRIGGERS = ["какие таймеры", "мои напоминания", "список напоминаний", "активные таймеры", "покажи напоминания", "какие есть напоминания"]
REMINDER_DELETE_TRIGGERS = ["удали таймер", "удали напоминание", "отмени напоминание", "убери таймер", "удали все напоминания", "отмени все таймеры"]

# Регулярка для приветствий (клубничка) и список эмодзи
GREETING_REGEX = re.compile(r'\b(привет|здравствуйте|хай|ку|доброе утро|добрый день|добрый вечер|приветствую|здорово|салют)\b', re.IGNORECASE)
GREETING_REACTIONS = ["🥰", "❤️", "🤝", "🍓"]

# --- ФОНОВЫЙ ТАЙМЕР НАПОМИНАНИЙ ---
async def reminder_worker():
    logging.info("Таймер напоминаний запущен.")
    while True:
        await asyncio.sleep(30) 
        try:
            now = get_moscow_datetime_obj()
            async with async_session() as session:
                result = await session.execute(select(Reminder).where(Reminder.remind_time <= now))
                reminders = result.scalars().all()
                for rem in reminders:
                    try:
                        formatted_rem = format_text(f"🔔 **Напоминаю:**\n{rem.text}")
                        await bot.send_message(chat_id=rem.chat_id, text=formatted_rem, parse_mode="HTML")
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
    is_code_request = any(trigger in text_lower for trigger in CODE_TRIGGERS) or "```" in text
    is_political_request = any(trigger in text_lower for trigger in POLITICAL_TRIGGERS)
    is_detail_request = any(trigger in text_lower for trigger in DETAIL_TRIGGERS)
    is_music_request = any(trigger in text_lower for trigger in MUSIC_TRIGGERS)
    
    is_reminder_create = any(trigger in text_lower for trigger in REMINDER_CREATE_TRIGGERS)
    is_reminder_list = any(trigger in text_lower for trigger in REMINDER_LIST_TRIGGERS)
    is_reminder_delete = any(trigger in text_lower for trigger in REMINDER_DELETE_TRIGGERS)

    if is_reminder_delete or is_reminder_list: is_reminder_create = False
    if is_political_request: 
        pass
    if is_music_request: 
        is_code_request = False
        is_political_request = False

    is_group = message.chat.type in ["group", "supergroup"]
    
    if is_group and not image_b64 and not (is_triggered or is_reply_to_bot or is_code_request or is_political_request or is_detail_request or is_reminder_create or is_reminder_list or is_reminder_delete or is_music_request):
        now = time.time()
        if chat_id in chat_join_cooldowns and now - chat_join_cooldowns[chat_id] < CHAT_JOIN_COOLDOWN: return
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
                    response = await client.chat.completions.create(model=agent["model"], messages=[{"role": "system", "content": agent["prompt"]}, {"role": "user", "content": prompt}], temperature=0.9, max_tokens=80)
                await send_long_message(message, safe_get_content(response))
            except Exception as e: logging.error(f"Ошибка вклинивания: {e}")
            finally: typing_task.cancel()
        return 

    now = time.time()
    if user_id in user_cooldowns and now - user_cooldowns[user_id] < COOLDOWN_SECONDS: return
    user_cooldowns[user_id] = now

    async with user_locks[key]:
        
        # --- МУЗЫКА ---
        if is_music_request:
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                extract_prompt = f"Из запроса пользователя вычлени только исполнителя и название песни для поиска. Верни ТОЛЬКО поисковый запрос. Запрос: '{text}'"
                async with api_semaphore:
                    extract_response = await client.chat.completions.create(model="deepseek-v4.1-flash", messages=[{"role": "user", "content": extract_prompt}], temperature=0.1, max_tokens=50)
                search_query = safe_get_content(extract_response).strip()
                track = await search_jamendo(search_query)
                if track and track.get("audio_url"):
                    await message.reply_audio(audio=track["audio_url"], title=track["name"], performer=track["artist"], caption=f"Нашел трек: **{track['name']}** — {track['artist']}")
                else:
                    await message.reply(f"Не удалось найти трек по запросу «{search_query}».")
            except Exception: await message.reply("Не удалось найти музыку.")
            finally: typing_task.cancel()
            return

        # --- НАПОМИНАНИЯ ---
        if is_reminder_list:
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                reminders = await get_active_reminders(chat_id, user_id)
                if not reminders: await message.reply("У тебя нет активных напоминаний.")
                else:
                    reply_text = "**Твои активные напоминания:**\n\n"
                    for i, rem in enumerate(reminders, 1):
                        reply_text += f"{i}. **{rem.remind_time.strftime('%d.%m.%Y в %H:%M')}** — {rem.text}\n"
                    await message.reply(format_text(reply_text), parse_mode="HTML")
            finally: typing_task.cancel()
            return

        if is_reminder_delete:
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                parse_prompt = f"Пользователь пишет: '{text}'. Он хочет удалить напоминание. Если хочет удалить ВСЕ, верни JSON: {{\"delete_all\": true}}. Если конкретное, выдели слово и верни JSON: {{\"query\": \"слово\"}}. Ответь СТРОГО JSON."
                async with api_semaphore:
                    parse_response = await client.chat.completions.create(model="gpt-6.1-sol", messages=[{"role": "user", "content": parse_prompt}], temperature=0.1, max_tokens=50)
                data = json.loads(re.sub(r'```json|```', '', safe_get_content(parse_response)).strip())
                if data.get("delete_all"):
                    await delete_reminders_by_query(chat_id, user_id, None)
                    await message.reply("Хорошо, я удалил **все** твои напоминания.")
                elif data.get("query"):
                    await delete_reminders_by_query(chat_id, user_id, data["query"])
                    await message.reply(f"Понял. Если было напоминание со словом **«{data['query']}»**, я его удалил.")
                else: await message.reply("Не понял, какое именно напоминание удалить.")
            except Exception: await message.reply("Не удалось удалить напоминание.")
            finally: typing_task.cancel()
            return

        if is_reminder_create:
            typing_task = asyncio.create_task(keep_typing(chat_id))
            try:
                current_time = get_moscow_time()
                parse_prompt = f"Текущее время: {current_time} (Москва). Пользователь просит: '{text}'. Выдели текст и время. Ответь JSON: {{\"text\": \"что\", \"datetime_str\": \"YYYY-MM-DD HH:MM\"}}. Если нет времени: {{\"error\": \"no_time\"}}"
                async with api_semaphore:
                    parse_response = await client.chat.completions.create(model="gpt-6.1-sol", messages=[{"role": "user", "content": parse_prompt}], temperature=0.1, max_tokens=100)
                data = json.loads(re.sub(r'```json|```', '', safe_get_content(parse_response)).strip())
                if "error" in data: await message.reply("Я не понял время. Уточни (например: 'через 10 минут')."); return
                rem_datetime = datetime.strptime(data["datetime_str"], "%Y-%m-%d %H:%M")
                if rem_datetime <= get_moscow_datetime_obj(): await message.reply("Это время уже прошло."); return
                async with async_session() as session:
                    session.add(Reminder(chat_id=chat_id, user_id=user_id, text=data["text"], remind_time=rem_datetime))
                    await session.commit()
                await message.reply(format_text(f"Хорошо, напомню **{rem_datetime.strftime('%d.%m.%Y в %H:%M')}**:\n_{data['text']}_"), parse_mode="HTML")
            except Exception: await message.reply("Ошибка создания напоминания.")
            finally: typing_task.cancel()
            return

        # --- ТЕКСТ ---
        history = await get_user_history(chat_id, user_id)
        typing_task = asyncio.create_task(keep_typing(chat_id))
        try:
            time_context = f"\n\nВажная информация: Текущая дата и время — {get_moscow_time()}."
            if is_code_request:
                agent = AGENTS["code"]
                final_text = f"Проанализируй этот код или ошибку:\n\n{text}"
                temp = 0.7
                active_prompt = agent["prompt"] + time_context
                max_tok = 1500
            elif is_political_request:
                agent = AGENTS["political"]
                search_results = await search_tavily(text)
                final_text = f"Тема: \"{text}\"\n\nДанные:\n---\n{search_results}\n---\nДай КРАТКИЙ ответ (1-3 предложения) ПРОСТЫМ языком. Если просит подробностей, тогда разверни анализ."
                temp = 0.7
                active_prompt = agent["prompt"] + time_context
                max_tok = 500
            else:
                agent = AGENTS["chat"]
                final_text = text
                temp = 0.85
                active_prompt = agent["prompt"] + time_context
                max_tok = 80 # ЖЕСТКОЕ ОГРАНИЧЕНИЕ ДЛИНЫ

            await save_user_message(chat_id, user_id, "user", final_text)
            safe_history = truncate_history(history[-4:]) 
            
            if image_b64:
                messages = [{"role": "system", "content": active_prompt}] + safe_history + [{"role": "user", "content": [{"type": "text", "text": final_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]}]
                model_to_use = "gpt-6.1-sol" 
            else:
                messages = [{"role": "system", "content": active_prompt}] + safe_history + [{"role": "user", "content": final_text}]
                model_to_use = agent["model"]

            async with api_semaphore:
                response = await client.chat.completions.create(
                    model=model_to_use, 
                    messages=messages, 
                    temperature=temp,
                    max_tokens=max_tok
                )
            
            bot_reply = safe_get_content(response)
            await send_long_message(message, bot_reply)
            await save_user_message(chat_id, user_id, "assistant", bot_reply)
            
        except Exception as e:
            error_str = str(e).lower()
            logging.exception("Ошибка API")
            if "content_policy" in error_str or "safety" in error_str or "400" in error_str:
                await message.reply("Ой, кажется, я перегнул палку с матом, и меня заблокировали фильтры безопасности. Давай сменим тему, а? 😅")
            elif "connection" in error_str or "timeout" in error_str:
                await message.reply("Ой, кажется, у меня пропала связь с сервером. Попробуй отправить сообщение еще раз через минуту.")
            else:
                await message.reply("Произошла ошибка при обработке запроса.")
        finally:
            typing_task.cancel()

# --- ОБРАБОТЧИКИ МЕДИА ---
@dp.message(F.voice)
async def handle_voice(message: types.Message):
    if not message.from_user: return
    user_id, chat_id = message.from_user.id, message.chat.id
    typing_task = asyncio.create_task(keep_typing(chat_id))
    try:
        file = await bot.get_file(message.voice.file_id)
        buffer = io.BytesIO()
        await bot.download_file(file.file_path, destination=buffer)
        buffer.name = "voice.ogg"
        buffer.seek(0)
        async with api_semaphore:
            transcript = await client.audio.transcriptions.create(model="whisper-large-v3-turbo", file=buffer)
        text = transcript.text
        typing_task.cancel()
        await message.reply(f"🎤 Распознано: {text}")
        if (chat_id, user_id) in message_buffers:
            message_buffers[(chat_id, user_id)]["text"] += "\n" + text
            if message_buffers[(chat_id, user_id)]["task"]: message_buffers[(chat_id, user_id)]["task"].cancel()
        else:
            message_buffers[(chat_id, user_id)] = {"text": text, "message": message, "is_reply": is_reply_to_bot_check(message), "is_forwarded": message.forward_origin is not None}
        message_buffers[(chat_id, user_id)]["task"] = asyncio.create_task(delayed_process((chat_id, user_id)))
    except Exception as e: await message.reply(f"Не удалось распознать голосовое.")
    finally:
        if not typing_task.cancelled(): typing_task.cancel()

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    if not message.from_user: return
    user_id, chat_id = message.from_user.id, message.chat.id
    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        buffer = io.BytesIO()
        await bot.download_file(file.file_path, destination=buffer)
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        if message.caption:
            await process_aggregated_text(message, message.caption, is_reply_to_bot_check(message), message.forward_origin is not None, image_b64=img_b64)
        else:
            async with api_semaphore:
                response = await client.chat.completions.create(model="gpt-6.1-sol", messages=[{"role": "system", "content": "Опиши картинку кратко."}, {"role": "user", "content": [{"type": "text", "text": "Что тут?"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}], temperature=0.5, max_tokens=100)
            await save_user_message(chat_id, user_id, "user", f"[Пользователь отправил фото. На нем: {safe_get_content(response)}]")
            try: await message.react([types.ReactionTypeEmoji(emoji="👀")])
            except: pass
    except Exception as e: await message.reply("Не удалось обработать фото.")

@dp.message(F.document)
async def handle_document(message: types.Message):
    if not message.from_user: return
    user_id, chat_id = message.from_user.id, message.chat.id
    document = message.document
    filename = document.file_name or "file.txt"
    if document.file_size and document.file_size > 1024 * 1024: await message.reply("Файл больше 1 МБ."); return
    if os.path.splitext(filename)[1].lower() not in ['.py', '.txt', '.js', '.ts', '.html', '.css', '.json', '.md', '.jpg', '.jpeg', '.png']: await message.reply("Я понимаю только код и текст."); return
    if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png']: await message.reply("Пришли картинку как обычное фото."); return

    typing_task = asyncio.create_task(keep_typing(chat_id))
    try:
        file = await bot.get_file(document.file_id)
        downloaded_file = await bot.download_file(file.file_path)
        code_text = decode_text(downloaded_file.read())
        if len(code_text) > MAX_FILE_LENGTH: await message.reply("Файл слишком длинный."); return
        
        if not message.caption:
            async with pending_files_locks[(chat_id, user_id)]:
                pending_files[(chat_id, user_id)] = {"text": code_text, "filename": filename, "timestamp": time.time(), "status": "pending"}
            await message.reply("Файл получен. Что с ним сделать? Напиши задачу.")
            return
        
        async with api_semaphore:
            response = await client.chat.completions.create(model="gpt-6.1-sol", messages=[{"role": "system", "content": AGENTS["code"]["prompt"] + f"\nТекущее время: {get_moscow_time()}"}, {"role": "user", "content": f"Файл '{filename}'. Задача: '{message.caption}'.\n\nКод:\n\n{code_text}"}], temperature=0.7, max_tokens=1500)
        bot_reply = safe_get_content(response)
        await send_long_message(message, bot_reply)
        await save_user_message(chat_id, user_id, "assistant", bot_reply)
    except Exception: await message.reply("Не удалось прочитать файл.")
    finally: typing_task.cancel()

# --- ТЕКСТ ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я — Нафаня. Могу ответить на вопросы, разобрать код, найти музыку, понять фото/голосовые и поставить напоминание. (Рисование отключено для экономии). Спрашивай.")

@dp.message(F.text)
async def handle_message(message: types.Message):
    if not message.text or not message.from_user: return
    user_id, chat_id = message.from_user.id, message.chat.id
    key = (chat_id, user_id)
    if message.text.startswith('/'): return

    # --- РАНДОМНАЯ РЕАКЦИЯ НА ПРИВЕТСТВИЯ ---
    if GREETING_REGEX.search(message.text):
        try:
            random_reaction = random.choice(GREETING_REACTIONS)
            await message.react([types.ReactionTypeEmoji(emoji=random_reaction)])
        except Exception as e:
            logging.warning(f"Не удалось поставить реакцию: {e}")
    # -----------------------------------------
    
    async with pending_files_locks[key]:
        if key in pending_files:
            file_data = pending_files[key]
            if time.time() - file_data["timestamp"] > FILE_TTL_SECONDS: pending_files.pop(key, None)
            elif file_data["status"] == "processing": return 
            else:
                file_data["status"] = "processing"
                typing_task = asyncio.create_task(keep_typing(chat_id))
                try:
                    async with api_semaphore:
                        response = await client.chat.completions.create(model="gpt-6.1-sol", messages=[{"role": "system", "content": AGENTS["code"]["prompt"] + f"\nТекущее время: {get_moscow_time()}"}, {"role": "user", "content": f"Файл '{file_data['filename']}'. Задача: '{message.text}'.\n\nКод:\n\n{file_data['text']}"}], temperature=0.7, max_tokens=1500)
                    await send_long_message(message, safe_get_content(response))
                    await save_user_message(chat_id, user_id, "assistant", safe_get_content(response))
                    pending_files.pop(key, None)
                except Exception: await message.reply("Ошибка при обработке файла.")
                finally: typing_task.cancel()
                return

    if key in message_buffers:
        message_buffers[key]["text"] += "\n" + message.text
        message_buffers[key]["is_reply"] = message_buffers[key]["is_reply"] or is_reply_to_bot_check(message)
        message_buffers[key]["is_forwarded"] = message_buffers[key]["is_forwarded"] or (message.forward_origin is not None)
        if message_buffers[key]["task"]: message_buffers[key]["task"].cancel()
    else:
        message_buffers[key] = {"text": message.text, "message": message, "is_reply": is_reply_to_bot_check(message), "is_forwarded": message.forward_origin is not None}
        
    message_buffers[key]["task"] = asyncio.create_task(delayed_process(key))

# --- ТОЧКА ВХОДА ---
async def main():
    await bot.delete_webhook(drop_pending_updates=False) 
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logging.info("База данных готова.")
    asyncio.create_task(cleanup_db())
    asyncio.create_task(reminder_worker())
    logging.info("Запуск бота...")
    try: await dp.start_polling(bot)
    except Exception: logging.exception("Критическая ошибка")
    finally:
        await client.close()
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
