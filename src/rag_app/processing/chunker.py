from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tiktoken
from ..config import settings
from ..logger import log


# --- Tokenizer ---
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    log.warning(
        "tiktoken encoder 'cl100k_base' not found. Using character count for token size."
    )
    tokenizer = None


def get_token_count(text: str) -> int:
    """Helper function to count tokens for logging and chunking decisions."""
    if tokenizer:
        return len(tokenizer.encode(text))
    return len(text)  # Fallback to character count


def chunk_elements(elements: List[Any], source_filename: str) -> List[Document]:
    """
    Chunks unstructured elements into LangChain Document objects, each with metadata.

    Args:
        elements (List[Any]): A list of unstructured elements.
        source_filename (str): The name of the source PDF file to be added to metadata.

    Returns:
        List[Document]: A list of Document objects, ready for embedding.
    """
    if not elements:
        log.warning("The list of elements to chunk is empty. Returning an empty list.")
        return []

    log.info(
        f"Initializing text splitter with chunk_size={settings.CHUNK_SIZE} "
        f"and chunk_overlap={settings.CHUNK_OVERLAP}."
    )
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    final_chunks = []
    header_based_sections = []
    current_section_texts = []

    # Step 1: Group elements into sections (same as before)
    log.info("Grouping unstructured elements into sections based on 'Header' category.")
    for el in elements:
        if el.category == "Header":
            if current_section_texts:
                section_text = "\n".join(current_section_texts).strip()
                if section_text:
                    header_based_sections.append(section_text)
            current_section_texts = [el.text]
        else:
            if isinstance(el.text, str):
                current_section_texts.append(el.text)
            else:
                log.debug(f"Skipping non-string element text of type {type(el.text)}.")

    if current_section_texts:
        section_text = "\n".join(current_section_texts).strip()
        if section_text:
            header_based_sections.append(section_text)

    log.info(f"Created {len(header_based_sections)} sections based on headers.")

    # Step 2: Create Document objects and split oversized sections
    log.info("Creating Document objects and splitting oversized sections...")
    for i, section in enumerate(header_based_sections):
        # 3. Create a metadata dictionary for each document
        metadata = {"source": source_filename}
        section_token_count = get_token_count(section)

        if section_token_count <= settings.CHUNK_SIZE:
            # 4. If the section is small enough, create a single Document object
            chunk = Document(page_content=section, metadata=metadata)
            final_chunks.append(chunk)
        else:
            log.debug(
                f"Section {i+1} is too large ({section_token_count} tokens). "
                f"Splitting into smaller Document chunks."
            )
            # 5. For large sections, use split_documents which preserves metadata
            # First, create a single Document for the whole section
            section_doc = Document(page_content=section, metadata=metadata)
            # Then, split it into smaller documents
            sub_chunks = text_splitter.split_documents([section_doc])
            final_chunks.extend(sub_chunks)

    log.info(f"Total Document chunks created: {len(final_chunks)}")
    log.debug(f"Sample Document chunk: {final_chunks[0] if final_chunks else 'N/A'}")
    return final_chunks
