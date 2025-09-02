import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from ragas.metrics import ContextRelevance
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.dataset_schema import SingleTurnSample


# ==============================================================================
# METRIC 1: DETERMINISTIC EVALUATION (LEXICAL RELEVANCE)
# ==============================================================================


def evaluate_retriever_deterministic(
    query: str, documents: List[str]
) -> Dict[str, Any]:
    """
    Evaluates retriever relevance using a deterministic keyword-based method (BM25).
    - Measures direct keyword overlap.
    - Fast, free, and 100% repeatable.
    - Lacks semantic understanding (e.g., cannot understand synonyms).
    """
    if not documents:
        return {"average_score": 0.0, "scored_documents": []}

    tokenized_corpus = [doc.lower().split(" ") for doc in documents]
    tokenized_query = query.lower().split(" ")

    bm25 = BM25Okapi(tokenized_corpus)
    doc_scores = bm25.get_scores(tokenized_query)
    print("BM25 Scores for each document:", doc_scores)
    average_score = sum(doc_scores) / len(doc_scores) if len(doc_scores) else 0.0

    return {
        "metric_name": "BM25 Lexical Score",
        "average_score": average_score,
        "scores": doc_scores,
    }


# ==============================================================================
# METRIC 2: NON-DETERMINISTIC EVALUATION (SEMANTIC RELEVANCE)
# ==============================================================================


async def evaluate_retriever_non_deterministic(
    query: str, documents: List[str]
) -> Dict[str, Any]:
    """
    Evaluates retriever relevance using a non-deterministic LLM-as-a-Judge method.
    - Measures deep semantic meaning and context.
    - Slower and requires an API call.
    - Can understand synonyms, context, and nuance.
    - Results can have minor variations.
    """
    """
    This function defines the asynchronous evaluation task.
    """
    print("Setting up the evaluation sample and LLM...")

    if not documents:
        return {"metric_name": "RAGAs Semantic Score", "average_score": 0.0}

    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=documents,
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
    evaluator_llm = LangchainLLMWrapper(llm)
    scorer = ContextRelevance(llm=evaluator_llm)
    score = await scorer.single_turn_ascore(sample)
    return {
        "metric_name": "RAGAs Semantic Score",
        "average_score": score,
    }


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":

    # --- 1. Load Data and Configure Judge LLM ---
    load_dotenv()

    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Google API key not found in .env file.")

    # Your RAG pipeline output data
    rag_output = {
        "query": "real life problem related to differential equations?",
        "source_documents": [
            {
                "page_content": "M. aths - 12\nsolution of a differential equation indicates the actual concept of a differential equation..."
            },
            {
                "page_content": "erential Equation: A solution of an equation in a single variable is anumber which satisfies the equation..."
            },
            {
                "page_content": "Differential Equations\nobtained. This means that the general solution represents a family of curves..."
            },
            {
                "page_content": "Maths - 12 (RES\n| Differectial Equations\nFind the solution curve... The rate of consumption of oil... The rate of infection of a disease... The rate of reaction to a drug... The rate of increase of the number of cellular phone subscribers... A particle moves along the x-axis..."
            },
            {
                "page_content": "Differential Equations\nThere is a wide variety of differential equations which occur in engineering applications..."
            },
        ],
    }
    user_query = rag_output["query"]
    retrieved_docs_content = [
        doc["page_content"] for doc in rag_output["source_documents"]
    ]

    print("--- Starting Retriever Evaluation ---")
    print(f'Query: "{user_query}"')
    print(f"Retrieved {len(retrieved_docs_content)} documents.\n")

    # --- 2. Run Both Evaluations ---
    deterministic_result = evaluate_retriever_deterministic(
        user_query, retrieved_docs_content
    )
    non_deterministic_result = asyncio.run(
        evaluate_retriever_non_deterministic(user_query, retrieved_docs_content)
    )

    # --- 3. Display and Interpret Results ---
    print("--- Deterministic Evaluation ---")
    print(f"Metric: {deterministic_result['metric_name']}")
    print(f"Average Score: {deterministic_result['average_score']:.4f}")
    print(
        "Interpretation: Measures direct keyword overlap. A low score means the retrieved docs lack the exact words from the query.\n"
    )

    print("--- Non-Deterministic Evaluation ---")
    print("non_deterministic_result :", non_deterministic_result)
    print(f"Metric: {non_deterministic_result['metric_name']}")
    print(f"Average Score: {non_deterministic_result['average_score']:.4f}")
    print(
        "Interpretation: Measures semantic relevance using an LLM. A low score means an expert judge deemed the docs irrelevant to the query's intent.\n"
    )

    # --- 4. Final Diagnosis ---
    print("--- Final Diagnosis ---")
    det_score = deterministic_result["average_score"]
    non_det_score = non_deterministic_result["average_score"]

    if non_det_score < 0.5:
        print(
            "Primary Issue: The retriever is failing to find semantically relevant documents."
        )
        print(
            "Recommendation: This is a critical issue. Focus on improving the embedding model, document chunking strategy, or implementing query transformations."
        )
    elif det_score < 0.2 and non_det_score > 0.7:
        print(
            "Primary Issue: The retriever finds semantically related docs that lack specific keywords."
        )
        print(
            "Recommendation: This is a common scenario. Implement a keyword-based re-ranker (like BM25) to boost documents with specific term matches."
        )
    else:
        print(
            "Overall Performance: The retriever is performing reasonably well on both semantic and lexical fronts."
        )
