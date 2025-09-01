from pydantic import BaseModel
from typing import List, Dict, Any


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    """The response model including the answer and source documents."""

    answer: str
    source_documents: List
