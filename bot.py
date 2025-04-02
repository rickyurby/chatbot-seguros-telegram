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

from langchain_text_splitters import CharacterTextSplitter

from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS

from langchain.chains.qa_with_sources import load_qa_with_sources_chain  # Cambiado

from langchain_community.chat_models import ChatOpenAI  # Cambiado



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

    logger.error(f"⚠️ Error global: {context.error}")

    if update and update.message:

        await update.message.reply_text("😔 Ocurrió un error procesando tu solicitud")



async def health_check(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text("✅ El bot está funcionando correctamente")

    logger.info("Health check realizado")



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

        llm = ChatOpenAI(openai_api_key=OPENAI_KEY, temperature=0)  # Cambiado

        chain = load_qa_with_sources_chain(llm, chain_type="stuff")  # Cambiado

        response = chain({"input_documents": docs, "question": update.message.text})  # Cambiado

        await update.message.reply_text(response['output_text'][:4000])  # Cambiado

    except Exception as e:

        logger.error(f"Error en mensaje: {e}")

        await update.message.reply_text(f"⚠️ Error: {str(e)}")



async def register_webhook(app: Application):

    try:

        webhook_url = f"https://{os.getenv('RENDER_APP_NAME')}.onrender.com/webhook"

        secret_token = os.getenv('WEBHOOK_SECRET')

        

        current_info = await app.bot.get_webhook_info()

        logger.info(f"ℹ️ Webhook actual: {current_info.url} | Pendientes: {current_info.pending_update_count}")

        

        if current_info.url != webhook_url:

            logger.info("🔄 Configurando nuevo webhook...")

            await asyncio.sleep(1)

            await app.bot.set_webhook(

                url=webhook_url,

                secret_token=secret_token,

                allowed_updates=Update.ALL_TYPES,

                drop_pending_updates=True

            )

            logger.info(f"✅ Webhook configurado en {webhook_url}")

        else:

            logger.info("ℹ️ Webhook ya estaba configurado correctamente")

            

    except Exception as e:

        logger.error(f"❌ Error crítico al configurar webhook: {str(e)}")

        raise



if __name__ == "__main__":

    try:

        port = int(os.environ.get("PORT", 10000))

        

        app = Application.builder() \

            .token(TOKEN) \

            .post_init(register_webhook) \

            .build()

        

        app.add_handler(CommandHandler("start", start))

        app.add_handler(CommandHandler("health", health_check))

        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        app.add_error_handler(error_handler)

        

        logger.info(f"🚀 Iniciando aplicación en puerto {port}...")

        

        app.run_webhook(

            listen="0.0.0.0",

            port=port,

            webhook_url=f"https://{os.getenv('RENDER_APP_NAME')}.onrender.com/webhook",

            secret_token=os.getenv('WEBHOOK_SECRET'),

            cert=None,

            drop_pending_updates=True

        )

    except Exception as e:

        logger.critical(f"💥 Error fatal: {e}")

        raise
