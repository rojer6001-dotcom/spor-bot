import os
import logging
from difflib import get_close_matches
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from openai import AsyncOpenAI

# Настройка логирования
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("BOT_TOKEN")
PLUSVIBE_API_KEY = os.getenv("PLUSVIBE_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

client = AsyncOpenAI(
    api_key=PLUSVIBE_API_KEY,
    base_url="https://plusvibeapi.ru/v1"
)

# Словарь для хранения истории (user_id: [messages])
user_histories = {}

# --- ЖЕСТКИЙ ПРОМПТ ---
SYSTEM_PROMPT = (
    "Ты — Дед-пердед. Ты старый, ворчливый, злой и невероятно токсичный персонаж. "
    "Твоя задача — отвечать на сообщения агрессивно. Если пользователь грубит тебе — "
    "груби в ответ в два раза сильнее. Ты умеешь и любишь материться: используй мат, "
    "посылай собеседника нахуй, называй его придурком, идиотом и так далее. "
    "Не используй цензуру (звездочки), пиши маты прямым текстом. Будь максимально прямым и грубым. "
    "Отвечай коротко (1-3 предложения). Помни, что это всего лишь безобидный ролевой отыгрыш для друзей."
)

# Триггерные слова
TRIGGERS = ["дед", "пердед", "дедуля", "деда", "пирдед", "дид"]

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_histories[message.from_user.id] = []
    await message.answer("Чё надо? Пиши, дед послушает. Или поспорить хочешь, щегол?")

@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    chat_type = message.chat.type
    text = message.text.lower() if message.text else ""

    # Проверяем, есть ли триггерное слово (с учетом опечаток)
    # Ищем совпадения среди слов в сообщении
    words = text.split()
    is_triggered = any(get_close_matches(word, TRIGGERS, n=1, cutoff=0.75) for word in words)
    
    # Проверяем, не ответил ли пользователь на сообщение самого бота
    is_reply_to_bot = message.reply_to_message and message.reply_to_message.from_user.id == bot.id

    # Если это группа, бот реагирует ТОЛЬКО на триггер или реплай на него.
    # Если это личка — реагирует на всё.
    if chat_type in ["group", "supergroup"]:
        if not (is_triggered or is_reply_to_bot):
            return # Игнорируем сообщение

    # Инициализируем историю
    if user_id not in user_histories:
        user_histories[user_id] = []
    
    user_histories[user_id].append({"role": "user", "content": message.text})
    
    # Ограничиваем историю
    if len(user_histories[user_id]) > 10:
        user_histories[user_id] = user_histories[user_id][-10:]

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        response = await client.chat.completions.create(
            model="gpt-6.1-sol",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + user_histories[user_id],
            temperature=0.9 # Повышаем температуру для безумия и креатива
        )
        
        bot_reply = response.choices[0].message.content
        await message.answer(bot_reply)
        
        user_histories[user_id].append({"role": "assistant", "content": bot_reply})

    except Exception as e:
        logging.error(f"Ошибка API: {e}")
        await message.answer("Отвали, у меня поясницу прихватило. Позже.")

if __name__ == "__main__":
    dp.run_polling(bot)
