from pathlib import Path

from utils.save_load_metadata import save_ingested_data
from ingestion import chunking, vectorization, document_loader
from config import DATA_DIR, INGESTED_DATA_DIR, CHUNK_SIZE, OVERLAP_SIZE


def ingestion():

    file_paths = document_loader.get_file_path(DATA_DIR)
    all_chunks = []
    print(f"Chunking with size {CHUNK_SIZE} and overlap {OVERLAP_SIZE}")
    for pdf_path in file_paths:
        chunks = chunking.chunking(
            pdf_path=pdf_path, chunk_size=CHUNK_SIZE, overlap_size=OVERLAP_SIZE
        )
        all_chunks.extend(chunks)
    chunks_vectors, vectorizer = vectorization.vectorize_chunks(all_chunks)

    save_ingested_data(
        all_chunks, chunks_vectors, vectorizer, output_dir=INGESTED_DATA_DIR
    )
    print(
        f"Successfully ingested data from {len(file_paths)} PDF files and saved to {INGESTED_DATA_DIR}"
    )
