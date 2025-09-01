from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from ..config import settings
from .vector_store import load_vector_store
from ..common import spy_on_chain
from ..logger import log


def bm25_re_ranker(chain_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes the output of the parallel retrieval step and re-ranks the documents
    using BM25 based on the user's question.
    """
    log.info("Applying BM25 re-ranking...")
    question = chain_input.get("question")
    documents = chain_input.get("documents")

    if not documents or not question:
        log.warning("No documents or question found for re-ranking. Skipping.")
        return chain_input

    # Extract page content for BM25
    corpus = [doc.page_content for doc in documents]
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
    tokenized_query = question.lower().split(" ")

    # Fit BM25 and get scores
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Combine documents with scores and sort
    scored_documents = zip(documents, scores)
    sorted_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)

    # Extract the sorted documents
    re_ranked_docs = [doc for doc, score in sorted_documents]

    # Update the 'documents' key in the chain's input dictionary
    chain_input["documents"] = re_ranked_docs
    log.info(f"Re-ranked {len(re_ranked_docs)} documents.")

    return chain_input


def format_docs(docs: List[Document]) -> str:
    """Formats the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain_with_sources(retriever):
    """
    Creates a RAG chain that returns a dictionary containing the answer
    and the retrieved source documents.
    """
    log.info("Creating RAG chain with sources...")

    try:
        prompt = PromptTemplate.from_template(settings.PROMPT_TEMPLATE)
        llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY,
        )
    except Exception as e:
        log.error(f"Failed to initialize LLM or PromptTemplate: {e}")
        raise

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["documents"])))
        | RunnableLambda(lambda x: spy_on_chain(x, "Data before Prompt"))
        | prompt
        | RunnableLambda(lambda x: spy_on_chain(x, "Final Prompt to LLM"))
        | llm
        | StrOutputParser()
    )

    final_chain = (
        RunnableParallel(
            {
                "documents": retriever,
                "question": RunnablePassthrough(),
            }
        )
        | RunnableLambda(lambda x: spy_on_chain(x, "After Parallel Retrieval"))
        # --- NEW: BM25 Re-ranking Step ---
        # This step takes the retrieved docs and question, re-ranks the docs,
        # and passes the updated dictionary to the next step.
        | RunnableLambda(bm25_re_ranker)
        | RunnableLambda(lambda x: spy_on_chain(x, "After BM25 Re-ranking"))
        # ----------------------------------
        | RunnablePassthrough.assign(answer=rag_chain_from_docs)
    )

    log.info("RAG chain created successfully.")
    return final_chain


# --- Global Instances ---
log.info("Initializing RAG components...")
try:
    vector_store = load_vector_store()
    # Note: Retrieve more documents initially to give the re-ranker more to work with.
    # For example, retrieve k=10, and the re-ranker can pass the top 3 to the LLM.
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    log.info("Vector store and retriever initialized successfully.")

    main_rag_chain = create_rag_chain_with_sources(retriever)
except Exception as e:
    log.error(f"Fatal error during RAG component initialization: {e}")
    raise
