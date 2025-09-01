from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from ..config import settings
from ..common import save_json
from ..logger import log


# --- Tokenizer ---
# This can be initialized once at the module level
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


def chunk_elements(elements: List[Any]) -> List[str]:
    """
    Chunks unstructured elements using a header-based grouping strategy
    and LangChain's RecursiveCharacterTextSplitter for oversized sections.
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

    # Step 1: Group elements into sections based on 'Header' boundaries
    log.info("Grouping unstructured elements into sections based on 'Header' category.")
    for el in elements:
        if el.category == "Header":
            if current_section_texts:
                section_text = "\n".join(current_section_texts).strip()
                if section_text:  # Avoid adding empty sections
                    header_based_sections.append(section_text)
            current_section_texts = [el.text]
        else:
            # It's good practice to ensure el.text is a string
            if isinstance(el.text, str):
                current_section_texts.append(el.text)
            else:
                log.debug(f"Skipping non-string element text of type {type(el.text)}.")

    # Add the final section after the loop
    if current_section_texts:
        section_text = "\n".join(current_section_texts).strip()
        if section_text:
            header_based_sections.append(section_text)

    log.info(f"Created {len(header_based_sections)} sections based on headers.")

    # Step 2: Use LangChain's splitter on each section
    log.info("Splitting oversized sections into smaller chunks...")
    for i, section in enumerate(header_based_sections):
        section_token_count = get_token_count(section)
        if section_token_count <= settings.CHUNK_SIZE:
            final_chunks.append(section)
        else:
            # This is great for debugging to see which sections are being split
            log.debug(
                f"Section {i+1} is too large ({section_token_count} tokens). "
                f"Splitting into smaller chunks."
            )
            sub_chunks = text_splitter.split_text(section)
            final_chunks.extend(sub_chunks)

    log.info(f"Total chunks created: {len(final_chunks)}")
    log.debug(f"Text chunks: {final_chunks}")
    return final_chunks
