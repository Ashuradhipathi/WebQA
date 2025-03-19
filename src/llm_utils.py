import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Ensure required environment variables are set.
os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
os.environ.setdefault("USER_AGENT", "WebQA/1.0")  # Default value; adjust as needed.

def init_llm():
    """Initialize the language model."""
    return init_chat_model("llama3-8b-8192", model_provider="groq")

def init_vector_store():
    """Initialize the in-memory vector store with embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return InMemoryVectorStore(embeddings)
