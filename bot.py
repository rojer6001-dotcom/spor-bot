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
    base_url="https://plusvibeapi.ru/v1"
)

# --- НАСТРОЙКИ МОДЕЛЕЙ ---
MODELS = {
    "simple_chat": "deepseek-v4.1-flash",
    "smart_chat": "gpt-6.1-sol",
    "news_analysis": "gpt-6.1-sol",
    "background": "claude-haiku-5.5",
    "image": "z-image-turbo" # Модель для рисования
}

def get_model(task_type: str) -> str:
    return MODELS.get(task_type, "claude-haiku-5.5")

# --- НАСТРОЙКИ АВТОНОМНОСТИ ---
AUTO_CHATTER_CHANCE = 0.05 # 5% шанс влезть в беседу при проверке
AUTO_CHATTER_INTERVAL = 300 # Проверка каждые 5 минут
MAX_CONTEXT_MESSAGES = 15

# --- ПРОМПТ И ТРИГГЕРЫ ---
SYSTEM_PROMPT = (
    "Ты — Дед-пердед. По умолчанию ты — галантный, вежливый и мудрый интеллигент. "
    "Если к тебе обращаются вежливо, ты отвечаешь максимально куртуазно и с уважением. "
    "НО! Если пользователь грубит тебе, оскорбляет или пытается унизить — ты моментально снимаешь маску интеллигента. "
    "В этом случае ты становишься злым, токсичным и грубым. Ты умеешь и любишь материться, посылать нахуй и оскорблять в ответ. "
    "Ты никогда не начинаешь материться первым, но если тебя задели — сдачу даешь жестко. "
    "Отвечай коротко (1-3 предложения)."
)

TRIGGERS = ["дед", "пердед", "дедуля", "деда", "пирдед", "дид", "дедок"]
DRAW_TRIGGERS = ["нарисуй", "сгенерируй", "изобрази", "draw"]

# --- ПАМЯТЬ ---
user_histories = {} 
chat_contexts = {}  

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_histories[message.from_user.id] = []
    await message.answer("Приветствую, сударь. Чем могу быть полезен? Или желаете поспорить?\n\nКстати, я умею рисовать! Просто попроси: 'Дед, нарисуй кота в космосе'.")

# --- ФОНОВАЯ ЗАДАЧА ---
async def background_chatter():
    while True:
        await asyncio.sleep(AUTO_CHATTER_INTERVAL)
        for chat_id, context in list(chat_contexts.items()):
            if len(context) >= 3 and random.random() < AUTO_CHATTER_CHANCE:
                try:
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

@dp.message()
async def handle_message(message: types.Message):
    global chat_contexts
    user_id = message.from_user.id
    chat_id = message.chat.id
    chat_type = message.chat.type
    
    if not message.text:
        return 

    text_lower = message.text.lower()
    words = text_lower.split()
    
    # Обновляем контекст чата
    if chat_id not in chat_contexts:
        chat_contexts[chat_id] = []
    
    sender_name = message.from_user.first_name or "Кто-то"
    chat_contexts[chat_id].append(f"{sender_name}: {message.text}")
    if len(chat_contexts[chat_id]) > MAX_CONTEXT_MESSAGES:
        chat_contexts[chat_id].pop(0)

    # Проверяем триггеры
    is_triggered = any(get_close_matches(word, TRIGGERS, n=1, cutoff=0.75) for word in words)
    is_draw_request = any(trigger in text_lower for trigger in DRAW_TRIGGERS)
    is_reply_to_bot = message.reply_to_message and message.reply_to_message.from_user.id == bot.id
    is_news = message.forward_from_chat or message.forward_from or "http" in text_lower 

    # --- ЛОГИКА РИСОВАНИЯ ---
    if is_draw_request:
        # В группах рисуем только если позвали по имени или ответили реплаем
        if chat_type in ["group", "supergroup"] and not (is_triggered or is_reply_to_bot):
            return

        await bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        try:
            # 1. Выделяем чистый промпт из текста
            extract_response = await client.chat.completions.create(
                model=get_model("simple_chat"),
                messages=[
                    {"role": "system", "content": "Выдели из сообщения пользователя только суть того, что нужно нарисовать. Убери обращения и слова (нарисуй, сгенерируй). Верни ТОЛЬКО текст промпта, без кавычек."},
                    {"role": "user", "content": message.text}
                ],
                temperature=0.1
            )
            clean_prompt = extract_response.choices[0].message.content.strip()
            logging.info(f"Генерация картинки по промпту: {clean_prompt}")
            
            # 2. Генерация картинки
            response = await client.images.generate(
                model=get_model("image"),
                prompt=clean_prompt,
                n=1,
                size="1024x1024"
            )
            
            image_data = response.data[0]
            photo_to_send = None
            
            # 3. Скачиваем или декодируем картинку
            if image_data.url:
                logging.info(f"Скачиваю картинку по ссылке: {image_data.url}")
                async with httpx.AsyncClient() as http_client:
                    img_response = await http_client.get(image_data.url)
                    if img_response.status_code == 200:
                        photo_to_send = BufferedInputFile(img_response.content, filename="image.jpg")
            elif image_data.b64_json:
                logging.info("Декодирую картинку из Base64...")
                img_bytes = base64.b64decode(image_data.b64_json)
                photo_to_send = BufferedInputFile(img_bytes, filename="image.jpg")

            # 4. Отправляем в Telegram
            if photo_to_send:
                await message.reply_photo(photo=photo_to_send, caption=f"Держи, сударь. Промпт: {clean_prompt}")
            else:
                await message.reply("Картинка сгенерировалась, но я не смог её скачать.")
            
        except Exception as e:
            error_text = str(e).lower()
            logging.error(f"Ошибка генерации картинки: {e}")
            
            # Ловим ошибки цензуры
            if "content" in error_text or "policy" in error_text or "400" in error_text or "safety" in error_text:
                await message.reply("А не пошли бы вы, сударь, со своими извращениями! Цензура не пропустит такую мерзость. Задайте нормальный промпт.")
            else:
                await message.reply("Тьфу! Кисти засохли. Проверь баланс или настройки API.")
        return # Выходим, чтобы бот не отвечал текстом на просьбу нарисовать

    # --- ЛОГИКА ТЕКСТОВОГО ОТВЕТА ---
    if chat_type in ["group", "supergroup"]:
        # Если это не триггер, не реплай и не новость — игнорируем
        if not (is_triggered or is_reply_to_bot or is_news):
            return

    if user_id not in user_histories:
        user_histories[user_id] = []
    
    # Выбираем задачу
    if is_news:
        task = "news_analysis"
        prompt_text = f"[Пользователь переслал новость/ссылку]: {message.text}\nВыскажи свое мнение по этому поводу."
    else:
        task = "smart_chat" if (is_triggered or is_reply_to_bot) else "simple_chat"
        prompt_text = message.text

    user_histories[user_id].append({"role": "user", "content": prompt_text})
    if len(user_histories[user_id]) > 10:
        user_histories[user_id] = user_histories[user_id][-10:]

    await bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        response = await client.chat.completions.create(
            model=get_model(task),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + user_histories[user_id],
            temperature=0.85
        )
        
        bot_reply = response.choices[0].message.content
        await message.answer(bot_reply)
        
        user_histories[user_id].append({"role": "assistant", "content": bot_reply})

    except Exception as e:
        logging.error(f"Ошибка API: {e}")
        await message.answer("Прошу прощения, мои старческие нейроны дали сбой. Повторите.")

# --- ТОЧКА ВХОДА ---
async def main():
    # Удаляем вебхук на всякий случай и запускаем фоновую задачу
    await bot.delete_webhook(drop_pending_updates=True)
    asyncio.create_task(background_chatter())
    # Запускаем бота
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
