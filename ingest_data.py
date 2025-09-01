import sys
import os
from src.rag_app.processing import loader, chunker
from src.rag_app.core import vector_store
from src.rag_app.logger import log


def main(pdf_dir: str):
    """
    Scans a directory for PDF files, processes each one, and ingests the combined
    content into a single vector store.
    """
    log.info(f"--- Starting Data Ingestion Pipeline for directory: {pdf_dir} ---")

    # 1. Find all PDF files in the specified directory
    try:
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            log.warning(f"No PDF files found in directory: {pdf_dir}")
            return
    except FileNotFoundError:
        log.error(f"Error: Directory not found at {pdf_dir}")
        sys.exit(1)

    log.info(f"Found {len(pdf_files)} PDF files to process: {pdf_files}")

    all_chunks = []
    # 2. Process each PDF file and aggregate the chunks
    for pdf_file in pdf_files:
        full_path = os.path.join(pdf_dir, pdf_file)
        log.info(f"--- Processing file: {pdf_file} ---")

        # Load and partition the PDF
        elements = loader.load_and_partition_pdf(full_path)

        # Chunk the elements
        chunks = chunker.chunk_elements(elements, source_filename=pdf_file)
        log.info(f"Created {len(chunks)} chunks from {pdf_file}")
        all_chunks.extend(chunks)

    # 3. Create and persist the vector store with all aggregated chunks
    if all_chunks:
        log.info(
            f"--- Creating vector store with a total of {len(all_chunks)} chunks ---"
        )
        vector_store.create_vector_store(all_chunks)
        log.info("--- Data Ingestion Complete ---")
    else:
        log.warning(
            "No chunks were generated from the PDF files. Vector store not created."
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        log.warning("Usage: python ingest_data.py <path_to_pdf_directory>")
        sys.exit(1)

    pdf_directory_path = sys.argv[1]
    if not os.path.isdir(pdf_directory_path):
        log.error(f"Error: Provided path is not a directory: {pdf_directory_path}")
        sys.exit(1)

    main(pdf_directory_path)
