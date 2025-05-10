import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Токен Telegram-бота
TELEGRAM_BOT_TOKEN = ""  # Замените на ваш токен

# Загружаем модель (выбираем Mistral-7B для примера)
model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"  # Модель Mistral для диалогов
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Настройки логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Функция для генерации ответа
def generate_response(message: str) -> str:
    # Преобразуем запрос в формат, который понимает модель
    inputs = tokenizer.encode(f"Пожалуйста, отвечай на русском: {message}", return_tensors="pt")

    # Генерация ответа с ограничением на длину
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Декодируем и возвращаем ответ
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот с нейросетью. Напиши мне что-нибудь!")

# Обработка сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    response = generate_response(user_text)
    await update.message.reply_text(response)

# Запуск бота
async def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    await application.run_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
