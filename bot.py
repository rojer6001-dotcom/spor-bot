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

# Получаем токены из переменных окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
PLUSVIBE_API_KEY = os.getenv("PLUSVIBE_API_KEY")

# Инициализация бота и клиента
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

client = AsyncOpenAI(
    api_key=PLUSVIBE_API_KEY,
    base_url="https://plusvibeapi.ru/v1",
    timeout=30.0 
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

# Создаем файл базы данных bot_memory.db в папке с ботом
# sqlite+aiosqlite:/// означает локальный файл
engine = create_async_engine("sqlite+aiosqlite:///bot_memory.db", echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
logging.info("База данных SQLite инициализирована.")

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С БД ---
async def get_user_history(user_id: int) -> list:
    async with async_session() as session:
        result = await session.execute(
            select(UserMessage).where(UserMessage.user_id == user_id).order_by(UserMessage.id.desc()).limit(10)
        )
        messages = result.scalars().all()
        # Переворачиваем в хронологический порядок
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
    "Твоя задача — проанализировать код, ошибку или технический вопрос, который прислал пользователь. "
    "Найди баги, укажи на плохой стиль, объясни, почему код не работает (если это ошибка), и дай рекомендации по исправлению. "
    "Ты можешь быть токсичным и грубым (использовать мат), если пользователь написал полную чушь, но твои технические советы должны быть точными и полезными. "
    "Если код хороший — похвали сквозь зубы. Если это просто вопрос 'умеешь ли ты кодить' — ответь с достоинством, что ты старый хакер."
)

TRIGGERS = ["дед", "пердед", "дедуля", "деда", "пирдед", "дид", "дедок"]
DRAW_TRIGGERS = ["нарисуй", "сгенерируй", "изобрази", "draw"]
CODE_TRIGGERS = ["ошибка", "баг", "traceback", "exception", "почему не работает", "исправь", "ревью", "код", "кодить", "программ", "def ", "import ", "```"]

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
                                    {"role": "system", "content": SYSTEM_PROMPT + " Сейчас ты видишь переписку в чате. Вмешайся в нее своим экспертным или ворчливым мнением (но без мата, если не грубят)."},
                                    {"role": "system", "content": f"Последние сообщения в чате:\n" + "\n".join(context[-5:])}
                                ],
                                temperature=0.9
                            )
                            reply = response.choices[0].message.content
                            await bot.send_message(chat_id=chat_id, text=reply)
                            logging.info(f"Бот сам влез в чат {chat_id}")
        except Exception as e:
            logging.error(f"Ошибка фонового чата: {e}")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Приветствую, сударь. Чем могу быть полезен? Или желаете поспорить?\n\nКстати, я умею рисовать! Просто попроси: 'Дед, нарисуй кота в космосе'.\n\nА еще я могу разобрать твой кривой код. Кидай его сюда, я поржу.")

@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    chat_type = message.chat.type
    
    if not message.text:
        return 

    text_lower = message.text.lower()
    words = text_lower.split()
    
    # Сохраняем контекст чата в БД
    sender_name = message.from_user.first_name or "Кто-то"
    await save_chat_message(chat_id, f"{sender_name}: {message.text}")

    is_triggered = any(get_close_matches(word, TRIGGERS, n=1, cutoff=0.75) for word in words)
    is_draw_request = any(trigger in text_lower for trigger in DRAW_TRIGGERS)
    is_reply_to_bot = message.reply_to_message and message.reply_to_message.from_user.id == bot.id
    is_news = message.forward_from_chat or message.forward_from or "http" in text_lower 
    is_code_request = any(trigger in text_lower for trigger in CODE_TRIGGERS) or "```" in message.text

    # --- ЛОГИКА РИСОВАНИЯ ---
    if is_draw_request:
        await bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        try:
            extract_response = await client.chat.completions.create(
                model=get_model("simple_chat"),
                messages=[
                    {"role": "system", "content": "Выдели из сообщения пользователя только суть того, что нужно нарисовать. Убери обращения и слова (нарисуй, сгенерируй). Верни ТОЛЬКО текст промпта, без кавычек. Если запрос нарушает правила безопасности, верни ровно одно слово: REFUSED."},
                    {"role": "user", "content": message.text}
                ],
                temperature=0.1
            )
            clean_prompt = extract_response.choices[0].message.content.strip()
            
            if "REFUSED" in clean_prompt.upper() or "не могу" in clean_prompt.lower() or "извините" in clean_prompt.lower() or "запрещ" in clean_prompt.lower():
                await message.reply("А не пошли бы вы, сударь, со своими извращениями! Цензура не пропустит такую мерзость. Задайте нормальный промпт.")
                return

            logging.info(f"Генерация картинки по промпту: {clean_prompt}")
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
                await message.reply("Картинка сгенерировалась, но я не смог её скачать. Наверное, интернет барахлит.")
            
        except Exception as e:
            error_text = str(e).lower()
            logging.error(f"Ошибка генерации картинки: {e}")
            if "content" in error_text or "policy" in error_text or "400" in error_text or "safety" in error_text or "invalid" in error_text:
                await message.reply("А не пошли бы вы, сударь, со своими извращениями! Цензура не пропустит такую мерзость. Задайте нормальный промпт.")
            else:
                await message.reply("Тьфу! Кисти засохли, холст порвался. Проверь баланс или настройки API, а меня не трогай.")
        return 

    # --- ЛОГИКА ТЕКСТОВОГО ОТВЕТА ---
    if chat_type in ["group", "supergroup"]:
        if not (is_triggered or is_reply_to_bot or is_news or is_code_request):
            return

    # Загружаем историю из БД
    history = await get_user_history(user_id)
    
    if is_code_request:
        task = "code_review"
        active_prompt = CODE_PROMPT
        prompt_text = message.text
        logging.info("Запуск анализа кода...")
    elif is_news:
        task = "news_analysis"
        active_prompt = SYSTEM_PROMPT
        prompt_text = f"[Пользователь переслал новость/ссылку]: {message.text}\nВыскажи свое мнение по этому поводу."
    else:
        task = "smart_chat" if (is_triggered or is_reply_to_bot) else "simple_chat"
        active_prompt = SYSTEM_PROMPT
        prompt_text = message.text

    # Сохраняем сообщение пользователя в БД
    await save_user_message(user_id, "user", prompt_text)
    
    # Добавляем в локальный список для текущего запроса
    history.append({"role": "user", "content": prompt_text})
    history = history[-10:] # Обрезаем, если слишком длинно

    await bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        response = await client.chat.completions.create(
            model=get_model(task),
            messages=[{"role": "system", "content": active_prompt}] + history,
            temperature=0.85
        )
        
        bot_reply = response.choices[0].message.content
        await message.answer(bot_reply)
        
        # Сохраняем ответ бота в БД
        await save_user_message(user_id, "assistant", bot_reply)

    except Exception as e:
        logging.error(f"Ошибка API: {e}")
        await message.answer("Прошу прощения, мои старческие нейроны дали сбой. Повторите.")

# --- ТОЧКА ВХОДА ---
async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    
    # Создаем таблицы в БД при запуске (если файла bot_memory.db еще нет)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logging.info("Таблицы в SQLite проверены/созданы.")

    asyncio.create_task(background_chatter())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
