from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from pypdf import PdfReader
import requests
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Configuración
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

PDF_URLS = [
    "https://drive.google.com/file/d/1AAqvlCYUVYxl5iRPTjCkRVcDRTFjpzq2/view?usp=drive_link",
    "https://drive.google.com/file/d/1pFMjFmS-xlj9awXfXc3qc8Dh0xNv13cx/view?usp=drive_link",
    "https://drive.google.com/file/d/1jviAI9BUkgVsb0dDQGvmZNXlFpVdMmxy/view?usp=drive_link"
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
        await update.message.reply_text(f"📄 Respuesta:\n{response[:4000]}...")  # Limita a 4000 caracteres
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()