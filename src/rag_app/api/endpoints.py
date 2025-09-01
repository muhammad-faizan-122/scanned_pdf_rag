from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from .models import QueryRequest, QueryResponse
from ..core.chain import main_rag_chain
from ..logger import log
from typing import Dict, Any

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Handles a user query by invoking the RAG chain.

    This endpoint is asynchronous and runs the synchronous RAG chain in a
    separate thread pool to avoid blocking the server's event loop.

    Args:
        request: A QueryRequest object containing the user's query string.

    Returns:
        A QueryResponse object containing the generated answer and source documents.

    Raises:
        HTTPException: 400 if the query is empty.
        HTTPException: 500 for any internal server errors.
    """
    query = request.query
    log.info(f"Received query: '{query}'")

    # 1. --- Input Validation ---
    # Ensure the query is not empty or just whitespace.
    if not query or not query.strip():
        log.warning("Received an empty or whitespace-only query.")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 2. --- Asynchronous Execution ---
        # Run the synchronous `invoke` method in a thread pool to avoid blocking
        # the main async event loop. This is crucial for performance.
        result: Dict[str, Any] = await run_in_threadpool(main_rag_chain.invoke, query)
        log.info(f"Successfully generated response for query: '{query}'")
        log.info(f"RAG Answer: {result.get('answer', '')}")
        log.debug(f"Full RAG Chain Result: {result}")
        return QueryResponse(
            answer=result.get("answer", ""),
            source_documents=result.get("documents", []),
        )

    except Exception as e:
        # 3. --- Enhanced Error Logging ---
        # Include the query that caused the error for better debugging context.
        log.error(f"Error processing query '{query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your query.",
        )
