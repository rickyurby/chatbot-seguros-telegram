from dotenv import load_dotenv
import os
import requests
import asyncio
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Configuración inicial
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

PDF_URLS = [
    "https://drive.google.com/uc?export=download&id=1AAqvlCYUVYxl5iRPTjCkRVcDRTFjpzq2",
    "https://drive.google.com/uc?export=download&id=1pFMjFmS-xlj9awXfXc3qc8Dh0xNv13cx",
    "https://drive.google.com/uc?export=download&id=1jviAI9BUkgVsb0dDQGvmZNXlFpVdMmxy"
]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('🤖 ¡Hola! Soy tu asistente de pólizas.')

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja todos los errores no capturados"""
    logger.error(f"⚠️ Error global: {context.error}")
    if update and update.message:
        await update.message.reply_text("😔 Ocurrió un error procesando tu solicitud")

def process_pdfs():
    texts = []
    for i, url in enumerate(PDF_URLS):
        try:
            logger.info(f"📄 Procesando PDF {i+1}/{len(PDF_URLS)}")
            response = requests.get(url, timeout=60)
            with open(f"temp_{i}.pdf", "wb") as f:
                f.write(response.content)
                
            reader = PdfReader(f"temp_{i}.pdf")
            for page in reader.pages[:20]:
                if text := page.extract_text():
                    texts.append(text)
                    
        except Exception as e:
            logger.error(f"❌ Error con PDF {url}: {e}")
        finally:
            if os.path.exists(f"temp_{i}.pdf"):
                os.remove(f"temp_{i}.pdf")
    
    if not texts:
        raise ValueError("No se pudo procesar ningún PDF")
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text("\n".join(texts))
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    return FAISS.from_texts(chunks, embeddings)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_chat_action("typing")
        knowledge_base = process_pdfs()
        docs = knowledge_base.similarity_search(update.message.text)
        llm = OpenAI(openai_api_key=OPENAI_KEY, temperature=0)
        response = load_qa_chain(llm, chain_type="stuff").run(
            input_documents=docs, 
            question=update.message.text
        )
        await update.message.reply_text(response[:4000])
    except Exception as e:
        logger.error(f"Error en mensaje: {e}")
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

async def register_webhook(app: Application):
    """Registra el webhook verificando primero si es necesario"""
    try:
        current_webhook = await app.bot.get_webhook_info()
        webhook_url = f"https://{os.getenv('RENDER_APP_NAME')}.onrender.com/{TOKEN}"
        
        # Solo registrar el webhook si la URL no coincide
        if current_webhook.url != webhook_url:
            await app.bot.set_webhook(
                url=webhook_url,
                secret_token=os.getenv('WEBHOOK_SECRET'),
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            logger.info(f"✅ Webhook registrado en: {webhook_url}")
        else:
            logger.info("ℹ️ Webhook ya está configurado correctamente, no se necesita registrar nuevamente.")
    except Exception as e:
        logger.error(f"❌ Error al registrar webhook: {e}")
        raise


async def main():
    """Función principal para iniciar la app y registrar webhook"""
    app = Application.builder().token(TOKEN).build()

    await register_webhook(app)  # Solo registrar si es necesario

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("🚀 Iniciando aplicación...")
    await app.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        webhook_url=f"https://{os.getenv('RENDER_APP_NAME')}.onrender.com/{TOKEN}",
        secret_token=os.getenv('WEBHOOK_SECRET')
    )

if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(main())  # Ejecutar sin interferir con event loop
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.create_task(main())  # Ejecutarlo sin bloquear
        loop.run_forever()  # Mantener el bot activo

