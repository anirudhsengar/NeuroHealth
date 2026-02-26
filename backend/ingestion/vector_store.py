import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# We use GitHub Models for embeddings but the interface is OpenAI compatible
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("Warning: GITHUB_TOKEN environment variable not set.")

def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=GITHUB_TOKEN,
        base_url="https://models.inference.ai.azure.com",
    )

def create_and_store_embeddings(file_path: str, persist_directory: str = "./chroma_db"):
    """Loads a markdown or text file, chunks it, embeds it, and saves to Chroma DB"""
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split text into manageable chunks for the LLM
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split document into {len(chunks)} chunks.")
    
    # Initialize the vector store using Github Models Embeddings
    embedding_model = get_embedding_model()
    
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=persist_directory
    )
    
    print(f"Successfully embedded and saved to {persist_directory}")
    return db

def get_retriever(persist_directory: str = "./chroma_db"):
    """Returns a retriever interface for the stored embeddings"""
    db = Chroma(
        persist_directory=persist_directory, 
        embedding_function=get_embedding_model()
    )
    return db.as_retriever(search_kwargs={"k": 3})
