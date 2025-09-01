from langchain_chroma import Chroma

# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from typing import List
from ..config import settings
from ..logger import log


def get_embedding_function():
    return OllamaEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
    # return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)


def create_vector_store(chunks: List[str]):
    log.debug(
        f"Creating vector store with {len(chunks)} chunks using {settings.EMBEDDING_MODEL_NAME} embeddings Model."
    )
    embedding_function = get_embedding_function()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=settings.DB_PERSIST_DIR,
    )
    return vector_store


def load_vector_store():
    log.debug(f"Loading {settings.EMBEDDING_MODEL_NAME} embeddings Model.")
    embedding_function = get_embedding_function()
    vector_store = Chroma(
        persist_directory=settings.DB_PERSIST_DIR,
        embedding_function=embedding_function,
    )
    return vector_store
