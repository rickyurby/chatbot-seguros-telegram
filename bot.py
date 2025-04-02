from dotenv import load_dotenv  # Nueva importación crítica
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

# Cargar variables de entorno
load_dotenv()  # Ahora funcionará correctamente
TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

PDF_URLS = [
    "https://drive.google.com/uc?export=download&id=1AAqvlCYUVYxl5iRPTjCkRVcDRTFjpzq2",
    "https://drive.google.com/uc?export=download&id=1pFMjFmS-xlj9awXfXc3qc8Dh0xNv13cx",
    "https://drive.google.com/uc?export=download&id=1jviAI9BUkgVsb0dDQGvmZNXlFpVdMmxy"
]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('🤖 ¡Hola! Soy tu asistente de pólizas. Pregúntame lo que necesites.')

def process_pdfs():
    texts = []
    for url in PDF_URLS:
        response = requests.get(url)
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        reader = PdfReader("temp.pdf")
        texts.extend(page.extract_text() for page in reader.pages if page.extract_text())
    
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
        await update.message.reply_text(f"📄 Respuesta:\n{response[:4000]}...")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Puerto para Render
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Configuración especial para Render
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=TOKEN,
        webhook_url=f"https://your-render-app-name.onrender.com/{TOKEN}"
    )