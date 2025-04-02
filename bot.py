import os
import requests
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, CallbackContext
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno (para el token de Telegram y OpenAI)
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Lista de URLs de los PDFs en Google Drive (públicos)
PDF_URLS = [
    "https://drive.google.com/file/d/1AAqvlCYUVYxl5iRPTjCkRVcDRTFjpzq2/view?usp=drive_link",
    "https://drive.google.com/file/d/1pFMjFmS-xlj9awXfXc3qc8Dh0xNv13cx/view?usp=drive_link",
    "https://drive.google.com/file/d/1jviAI9BUkgVsb0dDQGvmZNXlFpVdMmxy/view?usp=drive_link"
]

def download_pdfs():
    texts = []
    for url in PDF_URLS:
        response = requests.get(url)
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        pdf_reader = PdfReader("temp.pdf")
        for page in pdf_reader.pages:
            texts.append(page.extract_text())
    os.remove("temp.pdf")
    return texts

def setup_knowledge_base():
    texts = download_pdfs()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text("\n".join(texts))
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def start(update: Update, context: CallbackContext):
    update.message.reply_text('¡Hola! Soy tu asistente de pólizas. Envíame tu pregunta.')

def handle_message(update: Update, context: CallbackContext):
    user_question = update.message.text
    knowledge_base = setup_knowledge_base()
    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=user_question)
    update.message.reply_text(response)

def main():
    updater = Updater(TELEGRAM_TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(filters.TEXT & ~Filters.command, handle_message))
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()