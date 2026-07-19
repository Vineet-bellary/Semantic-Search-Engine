from pathlib import Path

from semantic_search_engine.ingestion import chunking, document_loader
from semantic_search_engine.ingestion.representation.embedding import EmbeddingModel
from semantic_search_engine.utils.save_load_metadata import save_ingested_data
from semantic_search_engine.config import (
    DATA_DIR,
    INGESTED_DATA_DIR,
    CHUNK_SIZE,
    OVERLAP_SIZE,
)


def ingestion():

    file_paths = document_loader.get_file_path(DATA_DIR)
    embedding_model = EmbeddingModel()
    all_chunks = []

    print(f"Chunking with size {CHUNK_SIZE} and overlap {OVERLAP_SIZE}")
    for pdf_path in file_paths:
        chunks = chunking.chunking(
            pdf_path=pdf_path, chunk_size=CHUNK_SIZE, overlap_size=OVERLAP_SIZE
        )
        all_chunks.extend(chunks)

    embeddings = embedding_model.embed_chunks(all_chunks)

    print(f"\n{len(all_chunks)} chunks created from {len(file_paths)} PDF files...")
    print(f"{embeddings.shape} embeddings generated for the chunks...\n")

    save_ingested_data(all_chunks, embeddings, output_dir=INGESTED_DATA_DIR)
    print(
        f"Successfully ingested data from {len(file_paths)} PDF files and saved to {INGESTED_DATA_DIR}"
    )
