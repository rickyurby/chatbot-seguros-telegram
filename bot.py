import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Configuración inicial
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

# URLs de los PDFs en Google Drive (reemplaza con tus IDs)
PDF_URLS = [
    "https://drive.google.com/uc?export=download&id=TU_ID_PDF_1",
    "https://drive.google.com/uc?export=download&id=TU_ID_PDF_2"
]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('🔍 Hola! Soy tu asistente de pólizas. Envíame tu pregunta sobre los documentos.')

def process_pdfs():
    """Descarga y procesa los PDFs para crear la base de conocimiento"""
    texts = []
    for url in PDF_URLS:
        response = requests.get(url)
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        reader = PdfReader("temp.pdf")
        for page in reader.pages:
            texts.append(page.extract_text())
    
    # Procesamiento con LangChain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text("\n".join(texts))
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_question = update.message.text
        knowledge_base = process_pdfs()
        
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI(openai_api_key=OPENAI_KEY, temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        
        await update.message.reply_text(f"📄 Respuesta:\n{response}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

def main():
    # Configuración del bot
    app = Application.builder().token(TOKEN).build()
    
    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Iniciar bot
    print("Bot en ejecución...")
    app.run_polling()

if __name__ == "__main__":
    main()