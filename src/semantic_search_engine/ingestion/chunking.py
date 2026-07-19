from pathlib import Path
from semantic_search_engine.config import CHUNK_SIZE, OVERLAP_SIZE
from semantic_search_engine.ingestion.text_extraction import extract_text_from_pdf


def chunking(
    pdf_path: Path, chunk_size: int | None = None, overlap_size: int | None = None
) -> list[dict]:
    chunk_size = chunk_size or CHUNK_SIZE
    overlap_size = overlap_size or OVERLAP_SIZE
    pdf_page_data = extract_text_from_pdf(pdf_path)

    chunks = []
    for page_data in pdf_page_data:
        text = page_data["text"]
        document_name = page_data["document_name"]
        page_number = page_data["page_number"]

        words = text.split()
        step = chunk_size - overlap_size

        for i in range(0, len(words), step):
            chunk = words[i : i + chunk_size]
            text_chunk = " ".join(chunk)
            chunk_index = len(chunks)
            document_id = pdf_path.stem.replace(" ", "_").lower()
            chunk_id = f"{document_id}_{page_number}_{chunk_index + 1}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_name": document_name,
                    "page_number": page_number,
                    "text_chunk": text_chunk,
                }
            )
    return chunks
