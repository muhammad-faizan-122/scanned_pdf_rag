from fastapi import FastAPI
from .api.endpoints import router as api_router
from .config import settings  # This will implicitly check for GOOGLE_API_KEY on startup

app = FastAPI(title="RAG Service API")

app.include_router(api_router, prefix="/api")


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the RAG Service API. Go to /docs for API documentation."
    }
