import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

load_dotenv()

PDF_PATH = "app/data/Scape-Community-Guidelines.pdf"
DB_PATH = "app/db"

def ingest_data():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    
    print(f"No. of pages : {len(docs)}")
    
    
    # chunking
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap = 150
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"No. of Chunks: {len(chunks)}")
    
    # Embedding data
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    
    # create connection to chroma
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = DB_PATH
    )
    
    print(f"Ingestion successful, saving to {DB_PATH}")
    
if __name__ == "__main__":
    ingest_data()
        