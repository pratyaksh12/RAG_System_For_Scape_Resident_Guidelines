# Scape Resident Guidelines RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer resident questions based on the Scape Community Guidelines PDF.

## Tech Stack
- **Framework**: LangChain
- **Vector Database**: ChromaDB
- **LLM & Embeddings**: OpenAI (GPT-4o, text-embedding-3-small)
- **Interface**: Streamlit

## Setup

1. **Install Dependencies**
   ```bash
   cd scape-rag-system
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configuration**
   Create a `.env` file in `scape-rag-system/` with your OpenAI API key:
   ```text
   OPENAI_API_KEY=sk-your-key-here
   ```

## Usage

### 1. Data Ingestion
Parses the PDF and populates the vector database. Run this whenever the PDF changes.
```bash
./venv/bin/python app/ingest.py
```

### 2. Run Application
Launches the Streamlit chat interface.
```bash
streamlit run main.py
```

## Project Structure
- `app/ingest.py`: PDF parsing and vector indexing (Chunk Size: 600, Overlap: 150).
- `app/rag_chain.py`: RAG retrieval logic and LLM chain.
- `main.py`: Streamlit frontend application.