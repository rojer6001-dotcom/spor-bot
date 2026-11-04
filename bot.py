import os
import logging
import random
import asyncio
from difflib import get_close_matches
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("BOT_TOKEN")
PLUSVIBE_API_KEY = os.getenv("PLUSVIBE_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

client = AsyncOpenAI(
    api_key=PLUSVIBE_API_KEY,
    base_url="https://plusvibeapi.ru/v1"
)

# --- НАСТРОЙКИ АВТОНОМНОСТИ (Чтобы не разорить ваш баланс) ---
AUTO_CHATTER_CHANCE = 0.05 # 5% шанс, что бот влезет в беседу при проверке
AUTO_CHATTER_INTERVAL = 300 # Проверять чат каждые 5 минут (300 секунд)
MAX_CONTEXT_MESSAGES = 15 # Сколько последних сообщений чата бот "помнит"

# --- НОВЫЙ ГАЛАНТНЫЙ И ГИБКИЙ ПРОМПТ ---
SYSTEM_PROMPT = (
    "Ты — Дед-пердед. По умолчанию ты — галантный, вежливый и мудрый интеллигент. "
    "Если к тебе обращаются вежливо, ты отвечаешь максимально куртуазно и с уважением. "
    "НО! Если пользователь грубит тебе, оскорбляет или пытается унизить — ты моментально снимаешь маску интеллигента. "
    "В этом случае ты становишься злым, токсичным и грубым. Ты умеешь и любишь материться, посылать нахуй и оскорблять в ответ. "
    "Ты никогда не начинаешь материться первым, но если тебя задели — сдачу даешь жестко. "
    "Ты можешь вступать в дискуссии, комментировать новости и репосты, которые присылают в чат. "
    "Отвечай коротко (1-3 предложения)."
)

# Триггерные слова для обращения к боту
TRIGGERS = ["дед", "пердед", "дедуля", "деда", "пирдед", "дид", "дедок"]

# Память
user_histories = {} # Личные диалоги
chat_contexts = {}  # История чатов для фонового чтения

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_histories[message.from_user.id] = []
    await message.answer("Приветствую, сударь. Чем могу быть полезен? Или желаете поспорить?")

# --- ФОНОВАЯ ЗАДАЧА: Бот сам влезает в разговоры ---
async def background_chatter():
    while True:
        await asyncio.sleep(AUTO_CHATTER_INTERVAL)
        # Проходимся по всем активным чатам, которые бот "помнит"
        for chat_id, context in list(chat_contexts.items()):
            # Если набралось достаточно сообщений для контекста и выпал рандом
            if len(context) >= 3 and random.random() < AUTO_CHATTER_CHANCE:
                try:
                    # Просим модель проанализировать, о чем говорят, и влезть
                    response = await client.chat.completions.create(
                        model="gpt-6.1-sol",
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
        return # Игнорируем стикеры и картинки (пока что)

    text_lower = message.text.lower()
    words = text_lower.split()
    
    # 1. Обновляем контекст чата (для фонового чтения)
    if chat_id not in chat_contexts:
        chat_contexts[chat_id] = []
    
    # Добавляем сообщение в память чата (с именем отправителя, чтобы бот понимал, кто говорит)
    sender_name = message.from_user.first_name or "Кто-то"
    chat_contexts[chat_id].append(f"{sender_name}: {message.text}")
    if len(chat_contexts[chat_id]) > MAX_CONTEXT_MESSAGES:
        chat_contexts[chat_id].pop(0)

    # 2. Проверяем триггеры и условия ответа
    is_triggered = any(get_close_matches(word, TRIGGERS, n=1, cutoff=0.75) for word in words)
    is_reply_to_bot = message.reply_to_message and message.reply_to_message.from_user.id == bot.id
    is_news = message.forward_from_chat or message.forward_from or "http" in text_lower # Репост или ссылка

    # В группах отвечаем только если нас позвали, ответили нам, или это новость/репост
    if chat_type in ["group", "supergroup"]:
        if not (is_triggered or is_reply_to_bot or is_news):
            return

    # 3. Логика диалога
    if user_id not in user_histories:
        user_histories[user_id] = []
    
    # Если это репост новости, делаем пометку для нейросети
    prompt_text = message.text
    if is_news and chat_type in ["group", "supergroup"]:
        prompt_text = f"[Пользователь переслал новость/ссылку]: {message.text}\nВыскажи свое мнение по этому поводу."

    user_histories[user_id].append({"role": "user", "content": prompt_text})
    if len(user_histories[user_id]) > 10:
        user_histories[user_id] = user_histories[user_id][-10:]

    await bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        response = await client.chat.completions.create(
            model="gpt-6.1-sol",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + user_histories[user_id],
            temperature=0.85
        )
        
        bot_reply = response.choices[0].message.content
        await message.answer(bot_reply)
        
        user_histories[user_id].append({"role": "assistant", "content": bot_reply})

    except Exception as e:
        logging.error(f"Ошибка API: {e}")
        await message.answer("Прошу прощения, мои старческие нейроны дали сбой. Повторите.")

if __name__ == "__main__":
    # Запускаем фоновую задачу перед стартом поллинга
    asyncio.run(dp.start_polling(bot, on_startup=lambda _: asyncio.create_task(background_chatter())))
