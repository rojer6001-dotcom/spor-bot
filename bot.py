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

# --- НАСТРОЙКИ МОДЕЛЕЙ ПОД ЗАДАЧИ ---
MODELS = {
    "simple_chat": "deepseek-v4.1-flash",  # Самая дешевая и быстрая (для болтовни)
    "smart_chat": "gpt-6.1-sol",           # Умная (для споров, триггеров)
    "news_analysis": "gpt-6.1-sol",        # Для анализа ссылок и новостей
    "background": "claude-haiku-5.5"       # Средняя (для фонового чтения чата)
}

def get_model(task_type: str) -> str:
    """Возвращает название модели в зависимости от задачи"""
    return MODELS.get(task_type, "claude-haiku-5.5") # По умолчанию haiku

# --- НАСТРОЙКИ АВТОНОМНОСТИ ---
AUTO_CHATTER_CHANCE = 0.05
AUTO_CHATTER_INTERVAL = 300
MAX_CONTEXT_MESSAGES = 15

SYSTEM_PROMPT = (
    "Ты — Дед-пердед. По умолчанию ты — галантный, вежливый и мудрый интеллигент. "
    "Если к тебе обращаются вежливо, ты отвечаешь максимально куртуазно и с уважением. "
    "НО! Если пользователь грубит тебе, оскорбляет или пытается унизить — ты моментально снимаешь маску интеллигента. "
    "В этом случае ты становишься злым, токсичным и грубым. Ты умеешь и любишь материться, посылать нахуй и оскорблять в ответ. "
    "Ты никогда не начинаешь материться первым, но если тебя задели — сдачу даешь жестко. "
    "Ты можешь вступать в дискуссии, комментировать новости и репосты. "
    "Отвечай коротко (1-3 предложения)."
)

TRIGGERS = ["дед", "пердед", "дедуля", "деда", "пирдед", "дид", "дедок"]

user_histories = {} 
chat_contexts = {}  

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_histories[message.from_user.id] = []
    # Тут используем дешевую модель для приветствия
    model = get_model("simple_chat")
    await message.answer("Приветствую, сударь. Чем могу быть полезен? Или желаете поспорить?")

async def background_chatter():
    while True:
        await asyncio.sleep(AUTO_CHATTER_INTERVAL)
        for chat_id, context in list(chat_contexts.items()):
            if len(context) >= 3 and random.random() < AUTO_CHATTER_CHANCE:
                try:
                    # Для фоновых задач используем дешевую/среднюю модель, чтобы не разориться
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
    
    if chat_id not in chat_contexts:
        chat_contexts[chat_id] = []
    
    sender_name = message.from_user.first_name or "Кто-то"
    chat_contexts[chat_id].append(f"{sender_name}: {message.text}")
    if len(chat_contexts[chat_id]) > MAX_CONTEXT_MESSAGES:
        chat_contexts[chat_id].pop(0)

    is_triggered = any(get_close_matches(word, TRIGGERS, n=1, cutoff=0.75) for word in words)
    is_reply_to_bot = message.reply_to_message and message.reply_to_message.from_user.id == bot.id
    is_news = message.forward_from_chat or message.forward_from or "http" in text_lower 

    if chat_type in ["group", "supergroup"]:
        if not (is_triggered or is_reply_to_bot or is_news):
            return

    if user_id not in user_histories:
        user_histories[user_id] = []
    
    # --- ВЫБОР МОДЕЛИ ПОД ЗАДАЧУ ---
    if is_news:
        task = "news_analysis"
        prompt_text = f"[Пользователь переслал новость/ссылку]: {message.text}\nВыскажи свое мнение по этому поводу."
    elif is_triggered or is_reply_to_bot:
        task = "smart_chat"
        prompt_text = message.text
    else:
        task = "simple_chat"
        prompt_text = message.text

    model_to_use = get_model(task)
    logging.info(f"Использую модель {model_to_use} для задачи {task}")

    user_histories[user_id].append({"role": "user", "content": prompt_text})
    if len(user_histories[user_id]) > 10:
        user_histories[user_id] = user_histories[user_id][-10:]

    await bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        response = await client.chat.completions.create(
            model=model_to_use, # <--- Подставляем нужную модель
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
    asyncio.run(dp.start_polling(bot, on_startup=lambda _: asyncio.create_task(background_chatter())))
