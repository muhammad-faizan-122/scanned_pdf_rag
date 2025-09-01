import sys
import os
from src.rag_app.processing import loader, chunker
from src.rag_app.core import vector_store
from src.rag_app.logger import log


def main(pdf_path: str):
    log.info("--- Starting Data Ingestion Pipeline ---")

    # 1. Load and partition the PDF
    elements = loader.load_and_partition_pdf(pdf_path)

    # 2. Chunk the elements
    chunks = chunker.chunk_elements(elements)

    # 3. Create and persist the vector store
    vector_store.create_vector_store(chunks)

    log.info("--- Data Ingestion Complete ---")


if __name__ == "__main__":
    # Example: python scripts/ingest_data.py data/documents/my_document.pdf
    if len(sys.argv) != 2:
        log.warning("Usage: python ingest_data.py <path_to_pdf>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    if not os.path.exists(pdf_file_path):
        log.error(f"Error: File not found at {pdf_file_path}")
        sys.exit(1)

    main(pdf_file_path)
