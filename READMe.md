# Scanned PDF RAG Microservice

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a simple microservice for performing Retrieval-Augmented Generation (RAG) on scanned PDF documents. It leverages state-of-the-art layout-aware parsing, semantic chunking, and powerful language models to provide accurate, context-aware answers with source attribution.

## Features

-   **High-Resolution PDF Parsing**: Utilizes the `unstructured` library with its `hi_res` strategy to accurately parse text, tables, and images from scanned documents by understanding their layout.
-   **Intelligent Semantic Chunking**: Goes beyond fixed-size chunks by first grouping text semantically based on document headers.
-   **Advanced Text Splitting**: For oversized semantic sections, it uses LangChain's `RecursiveCharacterTextSplitter` to create appropriately sized chunks while respecting sentence and paragraph boundaries.
-   **High-Performance Embeddings**: Employs the `nomic-embed-text:latest` model to generate vector embeddings for text chunks.
-   **Vector Storage**: Uses `ChromaDB` for efficient local storage and retrieval of document embeddings.
-   **LLM Integration**: Leverages Google's `gemini-1.5-flash` model via the LangChain framework for generating accurate answers.
-   **RESTful API**: Exposes a clean, asynchronous API built with `FastAPI` for querying the RAG system.
-   **Source Attribution**: The API response includes the retrieved source document chunks, allowing for verification and debugging of the generated answer.
-   **Production-Ready Structure**: Organized as a scalable Python package with clear separation of concerns (API, core logic, data processing), centralized configuration, and structured logging.

## Technology Stack

-   **Backend Framework**: FastAPI
-   **LLM Orchestration**: LangChain
-   **Document Parsing**: `unstructured.io`
-   **Vector Database**: ChromaDB
-   **Embedding Model**: `nomic-embed-text:latest` (via Ollama)
-   **Generative LLM**: Google Gemini (`gemini-1.5-flash`)
-   **Deployment**: Docker

## Project Structure

The project is organized for scalability and maintainability.

```
.
├── data/                 # All variable data (documents, DB, images)
│   ├── chroma_db/        # Persisted ChromaDB vector store
│   └── documents/        # Location for source PDF files
├── ingest_data.py        # Standalone script for the data ingestion pipeline
├── logs/                 # Application log files
├── requirements.txt      # Python dependencies
└── src/
    └── rag_app/          # Main Python package for the application
        ├── api/          # FastAPI endpoints and Pydantic models
        ├── core/         # Core RAG logic (chain, vector store)
        ├── processing/   # Document loading and chunking logic
        ├── config.py     # Centralized configuration using Pydantic
        ├── logger.py     # Structured logging setup
        └── main.py       # FastAPI application entry point
```

## Setup and Installation

Follow these steps to set up the project locally.

### 1. Prerequisites

-   Python 3.10+
-   `git`
-   System dependencies for `unstructured` (OCR and layout detection).

For Debian/Ubuntu-based systems, install the following:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libgl1-mesa-glx
```

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment. You can either use `venv` or `conda`.

venv
```bash
python -m venv venv
source venv/bin/activate
```
Or 

Conda
```
conda create -n pdf_rag_env python=3.13.5 -y
conda activate pdf_rag_env
```


### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory of the project and add your Google API key.

```
# .env

GOOGLE_API_KEY="AIzaSy...your...google...api...key...here"
```

## Usage

The application has two main modes of operation: data ingestion and querying via the API.

### 1. Data Ingestion

Place your scanned PDF files into the `data/documents/` directory. Then, run the ingestion script, pointing it to the file you want to process.

This script will:
1.  Parse the PDF using `unstructured`.
2.  Chunk the content.
3.  Generate embeddings.
4.  Store the embeddings in the ChromaDB vector store located at `data/chroma_db/`.

```bash
python ingest_data.py data/documents/your_document.pdf
```
You only need to run this step once for each new document you want to add to the knowledge base.

### 2. Running the API Server

Start the FastAPI server using `uvicorn`. The `--reload` flag is useful for development.

```bash
uvicorn src.rag_app.main:app --host 0.0.0.0 --port 8000 --reload
```
The server will now be running and accessible at `http://localhost:8000`.

### 3. Querying the API

You can interact with the API through its documentation, which is automatically generated and available at `http://localhost:8000/docs`, or by using a tool like `curl`.

#### Example `curl` Request:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "tell me about maple?"
}'
```

## API Endpoint

### `POST /api/query`

-   **Description**: Submits a query to the RAG pipeline and returns a generated answer along with the source documents used.
-   **Request Body**:
    ```json
    {
      "query": "string"
    }
    ```
-   **Success Response (200 OK)**:
    ```json
    {
      "answer": "The generated answer from the LLM.",
      "source_documents": [
        {
          "page_content": "The text content of the retrieved chunk.",
          "metadata": {
            "source": "path/to/document.pdf",
            "...": "..."
          }
        }
      ]
    }
    ```
-   **Error Responses**:
    -   `400 Bad Request`: If the query is empty.
    -   `500 Internal Server Error`: If an unexpected error occurs during processing.

## Future Improvements
- **Multi-PDFs**: Implement optimized pipeline for ingestion of multi-pdf files.
-   **Re-ranking**: Implement a re-ranking step (e.g., using a cross-encoder) after the initial retrieval to improve the relevance of documents sent to the LLM.
-   **Query Transformations**: Add techniques like multi-query retrieval or step-back prompting to handle more complex or ambiguous user questions.
-   **Hybrid Search**: Combine vector search with traditional keyword search (e.g., BM25) to improve retrieval accuracy for queries containing specific codes or acronyms.
-   **Evaluation**: Build a systematic evaluation pipeline using frameworks like Ragas or LangChain's evaluation tools to measure the quality of the RAG system.