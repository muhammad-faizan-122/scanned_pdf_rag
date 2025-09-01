from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    # Load .env file if it exists
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Models ---
    # EMBEDDING_MODEL_NAME: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_MODEL_NAME: str = "nomic-embed-text:latest"
    LLM_MODEL_NAME: str = "gemini-1.5-flash"
    GOOGLE_API_KEY: str

    # --- Vector Store ---
    DB_PERSIST_DIR: str = "data/chroma_db"

    # --- Text Chunking ---
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # --- Prompt Template ---
    PROMPT_TEMPLATE: str = """You are a helpful QA assistant. \
Your task is to answer the question based ONLY on the provided context.
If the answer to the question is not contained within the context, \
you must say: "I apologize, the information is not available in the provided context."

**Context:**
{context}

**Question:**
{question}

**Answer:**"""


settings = Settings()
