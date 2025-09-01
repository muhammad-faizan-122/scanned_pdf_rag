# src/rag_app/core/chain.py

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


def format_docs(docs):
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

    # This sub-chain processes the retrieved documents and generates an answer.
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["documents"])))
        | RunnableLambda(lambda x: spy_on_chain(x, "Data before Prompt"))
        | prompt
        | RunnableLambda(lambda x: spy_on_chain(x, "Final Prompt to LLM"))
        | llm
        | StrOutputParser()
    )

    # The final chain that orchestrates the entire process.
    # It takes a user's query and performs two parallel tasks:
    # 1. 'documents': Retrieves relevant documents from the vector store.
    # 2. 'question': Passes the original query through unchanged for use in the prompt.
    final_chain = (
        RunnableParallel(
            {
                "documents": retriever,
                "question": RunnablePassthrough(),
            }
        )
        | RunnableLambda(lambda x: spy_on_chain(x, "After Parallel Retrieval"))
        | RunnablePassthrough.assign(answer=rag_chain_from_docs)
    )

    log.info("RAG chain created successfully.")
    return final_chain


# --- Global Instances ---
# This code runs once when the module is imported.
log.info("Initializing RAG components...")
try:
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    log.info("Vector store and retriever initialized successfully.")

    # Create the main RAG chain instance that will be used by the API
    main_rag_chain = create_rag_chain_with_sources(retriever)
except Exception as e:
    log.error(f"Fatal error during RAG component initialization: {e}")
    # Depending on the application, you might want to exit or handle this gracefully.
    raise
