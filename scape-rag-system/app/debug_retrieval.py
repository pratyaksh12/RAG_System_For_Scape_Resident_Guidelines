from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "app/db"

def debug_search(query):
    print(f"🔎 Searching for: '{query}'")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    results = vector_store.similarity_search_with_score(query, k=10)
    
    print(f"\nFound {len(results)} chunks.\n")
    for i, (doc, score) in enumerate(results):
        print(f"[{i+1}]")
        print(f"Content: {doc.page_content[:200]}")
        print("-" * 50)

if __name__ == "__main__":
    debug_search("can I smoke indoor?")
