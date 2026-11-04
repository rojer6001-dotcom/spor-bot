import os
import logging
import random
import asyncio
import base64
import httpx
from difflib import get_close_matches
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import BufferedInputFile
from openai import AsyncOpenAI

# Импорты для базы данных
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Text, BigInteger, select

# Настройка логирования
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("BOT_TOKEN")
PLUSVIBE_API_KEY = os.getenv("PLUSVIBE_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

client = AsyncOpenAI(
    api_key=PLUSVIBE_API_KEY,
    base_url="https://plusvibeapi.ru/v1",
    timeout=60.0 # Увеличили таймаут, так как склеенный код может обрабатываться дольше
)

# --- НАСТРОЙКИ БАЗЫ ДАННЫХ (SQLite) ---
Base = declarative_base()

class UserMessage(Base):
    __tablename__ = "user_messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    role: Mapped[str] = mapped_column(String(50))
    content: Mapped[str] = mapped_column(Text)

class ChatContext(Base):
    __tablename__ = "chat_contexts"
    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    message_text: Mapped[str] = mapped_column(Text)

engine = create_async_engine("sqlite+aiosqlite:///bot_memory.db", echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С БД ---
async def get_user_history(user_id: int) -> list:
    async with async_session() as session:
        result = await session.execute(
            select(UserMessage).where(UserMessage.user_id == user_id).order_by(UserMessage.id.desc()).limit(10)
        )
        messages = result.scalars().all()
        return [{"role": m.role, "content": m.content} for m in reversed(messages)]

async def save_user_message(user_id: int, role: str, content: str):
    async with async_session() as session:
        session.add(UserMessage(user_id=user_id, role=role, content=content))
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

# --- НАСТРОЙКИ МОДЕЛЕЙ И ПРОМПТЫ ---
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

SYSTEM_PROMPT = (
    "Ты — Дед-пердед. По умолчанию ты — галантный, вежливый и мудрый интеллигент. "
    "Если к тебе обращаются вежливо, ты отвечаешь максимально куртуазно и с уважением. "
    "НО! Если пользователь грубит тебе, оскорбляет или пытается унизить — ты моментально снимаешь маску интеллигента. "
    "В этом случае ты становишься злым, токсичным и грубым. Ты умеешь и любишь материться, посылать нахуй и оскорблять в ответ. "
    "Ты никогда не начинаешь материться первым, но если тебя задели — сдачу даешь жестко. "
    "Отвечай коротко (1-3 предложения)."
)

CODE_PROMPT = (
    "Ты — Дед-пердед, но сейчас ты выступаешь в роли опытного, циничного и ворчливого Senior-разработчика. "
    "Твоя задача — проанализировать код, ошибку или технический вопрос. "
    "Найди баги, укажи на плохой стиль, объясни, почему код не работает. "
    "Ты можешь быть токсичным и грубым (использовать мат), но твои технические советы должны быть точными. "
    "Если код хороший — похвали сквозь зубы."
)

TRIGGERS = ["дед", "пердед", "дедуля", "деда", "пирдед", "дид", "дедок"]
DRAW_TRIGGERS = ["нарисуй", "сгенерируй", "изобрази", "draw"]
CODE_TRIGGERS = ["ошибка", "баг", "traceback", "exception", "почему не работает", "исправь", "ревью", "код", "кодить", "программ", "def ", "import ", "```"]

# --- МЕХАНИЗМ СКЛЕИВАНИЯ СООБЩЕНИЙ ---
message_buffers = {} # user_id: {"text": str, "message": types.Message, "task": asyncio.Task}

async def delayed_process(user_id: int):
    # Ждем 1.5 секунды, чтобы Телега успела прислать все куски
    await asyncio.sleep(1.5) 
    data = message_buffers.pop(user_id, None)
    if not data:
        return
    
    message = data["message"]
    full_text = data["text"]
    
    try:
        await process_aggregated_text(message, full_text)
    except Exception as e:
        logging.error(f"Ошибка при обработке склеенного сообщения: {e}")
        await message.reply("Тьфу! Даже мои старческие нейроны не выдержали такого объема бреда.")

async def process_aggregated_text(message: types.Message, text: str):
    user_id = message.from_user.id
    chat_id = message.chat.id
    chat_type = message.chat.type
    
    text_lower = text.lower()
    words = text_lower.split()
    
    await save_chat_message(chat_id, f"{message.from_user.first_name or 'Кто-то'}: {text}")

    is_triggered = any(get_close_matches(word, TRIGGERS, n=1, cutoff=0.75) for word in words)
    is_reply_to_bot = message.reply_to_message and message.reply_to_message.from_user.id == bot.id
    is_news = message.forward_from_chat or message.forward_from or "http" in text_lower 
    
    is_code_request = any(trigger in text_lower for trigger in CODE_TRIGGERS) or "```" in text
    is_draw_request = any(trigger in text_lower for trigger in DRAW_TRIGGERS) and not is_code_request

    # --- ЛОГИКА РИСОВАНИЯ ---
    if is_draw_request:
        await bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        try:
            extract_response = await client.chat.completions.create(
                model=get_model("simple_chat"),
                messages=[
                    {"role": "system", "content": "Выдели из сообщения пользователя только суть того, что нужно нарисовать. Убери обращения и слова (нарисуй, сгенерируй). Верни ТОЛЬКО текст промпта, без кавычек. Если запрос нарушает правила безопасности, верни ровно одно слово: REFUSED."},
                    {"role": "user", "content": text}
                ],
                temperature=0.1
            )
            clean_prompt = extract_response.choices[0].message.content.strip()
            
            if "REFUSED" in clean_prompt.upper() or "не могу" in clean_prompt.lower() or "извините" in clean_prompt.lower() or "запрещ" in clean_prompt.lower():
                await message.reply("А не пошли бы вы, сударь, со своими извращениями! Цензура не пропустит такую мерзость.")
                return

            response = await client.images.generate(
                model=get_model("image"),
                prompt=clean_prompt,
                n=1,
                size="1024x1024"
            )
            
            image_data = response.data[0]
            photo_to_send = None
            
            if image_data.url:
                async with httpx.AsyncClient() as http_client:
                    img_response = await http_client.get(image_data.url)
                    if img_response.status_code == 200:
                        photo_to_send = BufferedInputFile(img_response.content, filename="image.jpg")
            elif image_data.b64_json:
                img_bytes = base64.b64decode(image_data.b64_json)
                photo_to_send = BufferedInputFile(img_bytes, filename="image.jpg")

            if photo_to_send:
                await message.reply_photo(photo=photo_to_send, caption=f"Держи, сударь. Промпт: {clean_prompt}")
            else:
                await message.reply("Картинка сгенерировалась, но я не смог её скачать.")
            
        except Exception as e:
            error_text = str(e).lower()
            logging.error(f"Ошибка генерации картинки: {e}")
            if "content" in error_text or "policy" in error_text or "400" in error_text or "safety" in error_text or "invalid" in error_text:
                await message.reply("А не пошли бы вы, сударь, со своими извращениями! Цензура не пропустит такую мерзость.")
            else:
                await message.reply("Тьфу! Кисти засохли. Проверь баланс или настройки API.")
        return 

    # --- ЛОГИКА ТЕКСТОВОГО ОТВЕТА И АНАЛИЗА КОДА ---
    if chat_type in ["group", "supergroup"]:
        if not (is_triggered or is_reply_to_bot or is_news or is_code_request):
            return

    history = await get_user_history(user_id)
    
    if is_code_request:
        task = "code_review"
        active_prompt = CODE_PROMPT
        logging.info(f"Запуск анализа кода (длина {len(text)} символов)...")
    elif is_news:
        task = "news_analysis"
        active_prompt = SYSTEM_PROMPT
        text = f"[Пользователь переслал новость/ссылку]: {text}\nВыскажи свое мнение."
    else:
        task = "smart_chat" if (is_triggered or is_reply_to_bot) else "simple_chat"
        active_prompt = SYSTEM_PROMPT

    await save_user_message(user_id, "user", text)
    history.append({"role": "user", "content": text})
    history = history[-10:]

    await bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        response = await client.chat.completions.create(
            model=get_model(task),
            messages=[{"role": "system", "content": active_prompt}] + history,
            temperature=0.85
        )
        
        bot_reply = response.choices[0].message.content
        await message.answer(bot_reply)
        await save_user_message(user_id, "assistant", bot_reply)

    except Exception as e:
        logging.error(f"Ошибка API: {e}")
        await message.answer("Прошу прощения, сударь. Видимо, текст настолько огромен, что мои старческие нейросети не выдержали. Попробуйте прислать файлом.")

# --- ОБРАБОТЧИКИ ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Приветствую, сударь. Чем могу быть полезен? Или желаете поспорить?\n\nКстати, я умею рисовать! Просто попроси: 'Дед, нарисуй кота в космосе'.\n\nА еще я могу разобрать твой кривой код. Кидай его сюда, я поржу.")

@dp.message()
async def handle_message(message: types.Message):
    if not message.text:
        return
        
    user_id = message.from_user.id
    
    # Игнорируем команды, чтобы не буферизировать их
    if message.text.startswith('/'):
        return
        
    # Добавляем текст в буфер
    if user_id in message_buffers:
        message_buffers[user_id]["text"] += "\n" + message.text
        # Отменяем старый таймер, так как пришло новое сообщение
        if message_buffers[user_id]["task"]:
            message_buffers[user_id]["task"].cancel()
    else:
        # Если буфера нет, создаем новый. Сохраняем самое первое сообщение для ответа
        message_buffers[user_id] = {"text": message.text, "message": message}
        
    # Запускаем новый таймер
    task = asyncio.create_task(delayed_process(user_id))
    message_buffers[user_id]["task"] = task

# --- ФОНОВАЯ ЗАДАЧА ---
async def background_chatter():
    while True:
        await asyncio.sleep(AUTO_CHATTER_INTERVAL)
        try:
            async with async_session() as session:
                result = await session.execute(select(ChatContext.chat_id).distinct())
                chat_ids = result.scalars().all()
                
                for chat_id in chat_ids:
                    if random.random() < AUTO_CHATTER_CHANCE:
                        context = await get_chat_context(chat_id)
                        if len(context) >= 3:
                            model = get_model("background")
                            response = await client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": SYSTEM_PROMPT + " Сейчас ты видишь переписку в чате. Вмешайся в нее своим мнением (без мата, если не грубят)."},
                                    {"role": "system", "content": f"Последние сообщения в чате:\n" + "\n".join(context[-5:])}
                                ],
                                temperature=0.9
                            )
                            reply = response.choices[0].message.content
                            await bot.send_message(chat_id=chat_id, text=reply)
        except Exception as e:
            logging.error(f"Ошибка фонового чата: {e}")

# --- ТОЧКА ВХОДА ---
async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logging.info("Таблицы в SQLite проверены/созданы.")

    asyncio.create_task(background_chatter())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
