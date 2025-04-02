from dotenv import load_dotenv
import os
import requests
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

def process_pdfs():
    texts = []
    for url in PDF_URLS:
        try:
            # Añade timeout y verificación SSL
            response = requests.get(url, timeout=30, verify=True)
            response.raise_for_status()
            
            # Procesamiento más rápido con chunks
            with open("temp.pdf", "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
                    
            reader = PdfReader("temp.pdf")
            texts.extend(page.extract_text() or "" for page in reader.pages[:50])  # Limita páginas
        except Exception as e:
            print(f"Error procesando {url}: {str(e)}")
            continue
        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")
    
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
        knowledge_base = process_pdfs()
        docs = knowledge_base.similarity_search(update.message.text)
        llm = OpenAI(openai_api_key=OPENAI_KEY, temperature=0)
        response = load_qa_chain(llm, chain_type="stuff").run(
            input_documents=docs, 
            question=update.message.text
        )
        await update.message.reply_text(response[:4000])
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Configuración definitiva para Render (CORREGIDA)
    port = int(os.environ.get("PORT", 5000))
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        webhook_url=f"https://{os.getenv('RENDER_APP_NAME')}.onrender.com/{TOKEN}",
        secret_token=os.getenv('WEBHOOK_SECRET'),
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES  # Este parámetro sí es válido
    )
