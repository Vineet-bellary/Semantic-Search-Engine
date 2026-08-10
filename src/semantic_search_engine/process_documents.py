from semantic_search_engine.utils import document_loader
from semantic_search_engine.ingestion.adapters.hybrid_chunk_adapter import (
    HybridChunkAdapter,
)
from semantic_search_engine.ingestion.chunkers.hybrid_chunker import hybrid_chunk
from semantic_search_engine.ingestion.encoders.embedding import EmbeddingModel
from semantic_search_engine.ingestion.parsers.doc_to_doclingobj import (
    configure_converter,
    parse_doc,
)
from semantic_search_engine.utils.save_load_metadata import save_ingested_data
from semantic_search_engine.config import (
    DATA_DIR,
    INGESTED_DATA_DIR,
)


def ingestion():
    """Ingest PDF documents from the data directory, convert them to Markdown, extract sections, create chunks, generate embeddings, and save the ingested data."""

    file_paths = document_loader.get_file_path(DATA_DIR, file_types={".pdf"})
    document_converter = configure_converter()
    chunk_adapter = HybridChunkAdapter()
    embedding_model = EmbeddingModel()
    all_chunks = []
    print(f"\nIngesting data from {len(file_paths)} PDF files in {DATA_DIR}...\n")
    for pdf_path in file_paths:
        doc = parse_doc(pdf_path, document_converter)
        raw_chunks = hybrid_chunk(doc)
        adapted_chunks = chunk_adapter.extract_info_chunk(raw_chunks)
        all_chunks.extend(adapted_chunks)

    embeddings = embedding_model.embed_chunks(all_chunks)

    print(f"\n{len(all_chunks)} chunks created from {len(file_paths)} PDF files...")
    print(f"{embeddings.shape} embeddings generated for the chunks...\n")

    save_ingested_data(all_chunks, embeddings, output_dir=INGESTED_DATA_DIR)
    print(
        f"\nSuccessfully ingested data from {len(file_paths)} PDF files and saved to {INGESTED_DATA_DIR}"
    )


if __name__ == "__main__":
    ingestion()
