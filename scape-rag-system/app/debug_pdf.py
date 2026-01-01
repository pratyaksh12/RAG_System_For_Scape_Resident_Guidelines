from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "app/data/Scape-Community-Guidelines.pdf"
OUTPUT_PATH = "app/data/extracted_text.txt"

def extract_to_text():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            f.write(f"--- PAGE {i+1} ---\n")
            f.write(doc.page_content)
            f.write("\n\n")
            

if __name__ == "__main__":
    extract_to_text()
