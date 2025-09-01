from unstructured.partition.pdf import partition_pdf
from typing import List, Any
from ..common import save_json
from ..logger import log


def save_elements_stats(elements: List[Any]) -> None:
    """Prints a summary of the elements extracted from the PDF."""
    if not elements:
        log.info("No elements found.")
        return

    log.debug(f"Total elements extracted: {len(elements)}")
    category_counts = {}
    for el in elements:
        category_counts[el.category] = category_counts.get(el.category, 0) + 1

    log.debug(f"Element category counts: {category_counts}")


def load_and_partition_pdf(pdf_path: str) -> List[Any]:
    """Loads a PDF and partitions it into elements using unstructured."""
    log.info(f"Partitioning PDF: {pdf_path}...")
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,  # mandatory to set as ``True``
        extract_image_block_types=["Image", "Table"],  # optional
        extract_image_block_to_payload=False,  # optional
        languages=["eng"],
        extract_image_block_output_dir="data/images",  # optional - only works when `extract_image_block_to_payload=False`
    )
    save_elements_stats(elements)
    log.info("PDF partitioning complete.")
    return elements
