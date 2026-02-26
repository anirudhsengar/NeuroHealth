"""
Ingestion script: Loads mock_guidelines.md into ChromaDB for RAG retrieval.
Run once: python ingest.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

from ingestion.vector_store import create_and_store_embeddings

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "mock_guidelines.md")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

if __name__ == "__main__":
    print(f"Ingesting: {DATA_FILE}")
    create_and_store_embeddings(DATA_FILE, PERSIST_DIR)
    print("Done! ChromaDB is now populated.")
