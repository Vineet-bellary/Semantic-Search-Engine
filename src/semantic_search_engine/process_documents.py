from pathlib import Path
from docling.document_converter import DocumentConverter


from semantic_search_engine.ingestion import (
    doc_to_markdown,
    document_loader,
    markdown_parser,
    retrieval_chunker,
)
from semantic_search_engine.ingestion.representation.embedding import EmbeddingModel
from semantic_search_engine.utils.save_load_metadata import save_ingested_data
from semantic_search_engine.config import (
    DATA_DIR,
    INGESTED_DATA_DIR,
)


def ingestion():
    """Ingest PDF documents from the data directory, convert them to Markdown, extract sections, create chunks, generate embeddings, and save the ingested data."""

    file_paths = document_loader.get_file_path(DATA_DIR, file_types={".pdf"})
    markdown_dir = Path(DATA_DIR) / "markdown"
    embedding_model = EmbeddingModel()
    document_converter = DocumentConverter()
    all_chunks = []
    print(f"\nIngesting data from {len(file_paths)} PDF files in {DATA_DIR}...\n")
    for pdf_path in file_paths:
        pdf_to_markdown = doc_to_markdown.pdf_to_markdown(
            pdf_path, markdown_dir, document_converter
        )
        sections = markdown_parser.extract_sections_from_markdown(pdf_to_markdown)
        chunks = retrieval_chunker.create_chunks(sections, pdf_path)
        all_chunks.extend(chunks)

    embeddings = embedding_model.embed_chunks(all_chunks)

    print(f"\n{len(all_chunks)} chunks created from {len(file_paths)} PDF files...")
    print(f"{embeddings.shape} embeddings generated for the chunks...\n")

    save_ingested_data(all_chunks, embeddings, output_dir=INGESTED_DATA_DIR)
    print(
        f"\nSuccessfully ingested data from {len(file_paths)} PDF files and saved to {INGESTED_DATA_DIR}"
    )


if __name__ == "__main__":
    ingestion()
