import json
import pickle
from pathlib import Path


def save_ingested_data(
    chunks: list[dict],
    chunks_vectors,
    vectorizer,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)

    vectors_path = output_dir / "tfidf_matrix_of_chunks.pkl"
    with open(vectors_path, "wb") as f:
        pickle.dump(chunks_vectors, f)

    vectorizer_path = output_dir / "vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Successfully saved all ingestion assets to {output_dir}")

def load_ingested_data(ingested_data_dir: Path):
    with open(ingested_data_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(ingested_data_dir / "tfidf_matrix_of_chunks.pkl", "rb") as f:
        chunks_vectors = pickle.load(f)

    with open(ingested_data_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
        
    if chunks is None or chunks_vectors is None or vectorizer is None:
        raise ValueError(f"One or more of the loaded assets are None. Please check the files in {ingested_data_dir}.")
    else:
        print(f"Successfully loaded all ingestion assets from {ingested_data_dir}")
    
    return chunks, chunks_vectors, vectorizer
