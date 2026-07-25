from pathlib import Path


def create_chunks(
    sections: list[dict[str, list[str] | str]], doc_path: Path
) -> list[dict[str, int | str | list[str]]]:
    """
    Create chunks from sections based on headings.

    Args:
        sections (list[dict[str, list[str] | str]]): List of sections extracted from a Markdown file.
        doc_path (Path): Path to the Markdown file.
    Returns:
        list[dict[str, int | str | list[str]]]: List of chunks created from the sections.
    """
    chunks = []

    for section in sections:
        heading_path = section["heading_path"]
        content = section["content"]

        # Create a chunk for the current section
        chunk_id = len(chunks)
        doc_name = doc_path.stem.replace(" ", "_").lower()
        chunk = {
            "chunk_id": chunk_id,
            "document_name": doc_name,
            "heading_path": heading_path,
            "text_chunk": content,
        }
        chunks.append(chunk)

    return chunks
