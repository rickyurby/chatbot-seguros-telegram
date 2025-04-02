import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv('TELEGRAM_TOKEN')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('¡Bot simplificado!')

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(update.message.text)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"⚠️ Error global: {context.error}")

async def health_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ El bot está funcionando correctamente")
    logger.info("Health check realizado")

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 10000))
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("echo", echo))
        app.add_handler(CommandHandler("health", health_check))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo)) #añadido el handler para responder a cualquier texto enviado.
        app.add_error_handler(error_handler)
        logger.info(f" Iniciando aplicación en puerto {port}...")
        app.run_webhook(listen="0.0.0.0", port=port, webhook_url=f"https://{os.getenv('RENDER_APP_NAME')}.onrender.com/webhook")
    except Exception as e:
        logger.critical(f" Error fatal: {e}")
        raise
