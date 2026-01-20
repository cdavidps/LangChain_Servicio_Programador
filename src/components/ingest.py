import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # <--- Cambio aquí
from langchain_community.vectorstores import Chroma
import chromadb

load_dotenv()

def run_ingestion():
    raw_data_path = "data/raw/services.txt"
    persist_db_path = "data/chroma_db"

    print(f"--- Iniciando ingesta local desde: {raw_data_path} ---")

    loader = TextLoader(raw_data_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # USAMOS UN MODELO LOCAL: No requiere API Key, corre en tu CPU
    # Es uno de los más usados en la industria para RAG ligero.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    try:
        # Desactivamos telemetría para limpiar la consola
        client_settings = chromadb.config.Settings(anonymized_telemetry=False)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_db_path,
            client_settings=client_settings
        )
        print(f"✅ Éxito: Base de datos vectorial persistida en {persist_db_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_ingestion()