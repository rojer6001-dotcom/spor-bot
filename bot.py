import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from openai import AsyncOpenAI

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Берем ключи из переменных окружения BotHost
BOT_TOKEN = os.getenv("BOT_TOKEN")
PLUSVIBE_API_KEY = os.getenv("PLUSVIBE_API_KEY")

# Инициализация
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

client = AsyncOpenAI(
    api_key=PLUSVIBE_API_KEY,
    base_url="https://plusvibeapi.ru/v1"
)

# Словарь для хранения истории (user_id: [messages])
user_histories = {}

# Настройки бота
SYSTEM_PROMPT = "Ты — уверенный в себе собеседник. Твоя цель — спорить, оспаривать аргументы, быть дерзким, но оставаться в рамках приличий. Отвечай коротко (до 3-4 предложений)."

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_histories[message.from_user.id] = []
    await message.answer("Привет! Готов поспорить с тобой на любую тему. Напиши свой тезис, и я его разнесу.")

@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    
    # Инициализируем историю, если ее нет
    if user_id not in user_histories:
        user_histories[user_id] = []
    
    # Добавляем сообщение пользователя в историю
    user_histories[user_id].append({"role": "user", "content": message.text})
    
    # Ограничиваем историю последними 10 сообщениями (для экономии токенов)
    if len(user_histories[user_id]) > 10:
        user_histories[user_id] = user_histories[user_id][-10:]

    # Отправляем "печатает..."
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        # Запрос к PlusVibe API с новой моделью
        response = await client.chat.completions.create(
            model="gpt-6.1-sol", # <--- НОВАЯ МОДЕЛЬ
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + user_histories[user_id],
            temperature=0.8
        )
        
        bot_reply = response.choices[0].message.content
        await message.answer(bot_reply)
        
        # Сохраняем ответ бота в историю
        user_histories[user_id].append({"role": "assistant", "content": bot_reply})

        # --- ЛОГИКА РЕКЛАМЫ ВРЕМЕННО ОТКЛЮЧЕНА ---
        # if AD_TEXT:
        #     ...

    except Exception as e:
        logging.error(f"Ошибка API: {e}")
        await message.answer("Что-то у меня в голове замкнуло. Давай еще раз.")

if __name__ == "__main__":
    dp.run_polling(bot)
