from pathlib import Path

from utils.save_load_metadata import load_ingested_data
from retrieval.process_query import preprocess_query, validate_query, vectorize_query
from retrieval.query import get_query
from retrieval.similarity import rank_chunks
from config import INGESTED_DATA_DIR, CONFIDENCE_THRESHOLD


def prepare_query(query: str, vectorizer) -> list[float]:
    preprocessed_query = preprocess_query(query)
    query_vector = vectorize_query(preprocessed_query, vectorizer)
    return query_vector


def search():
    chunks, chunks_vectors, vectorizer = load_ingested_data(INGESTED_DATA_DIR)

    if validate_query(query := get_query()):
        query_vector = prepare_query(query, vectorizer)

    scores, top_k_indices = rank_chunks(query_vector, chunks_vectors, num_suggestions=3)
    print(f"\n{'-' * 100}\nRelevant data found from your documents:\n{'-' * 100}\n")
    for sl_no, idx in enumerate(top_k_indices):
        # chunk =  chunk_id, document_name, page_number, chunk_text
        if scores[idx] < CONFIDENCE_THRESHOLD:
            continue

        chunk = chunks[idx]
        suggestion = f"Document Name: {chunk['document_name']}\nPage Number: {chunk['page_number']}\nScore: {scores[idx]:.2f}\n\nChunk Text:\n{chunk['text_chunk']}\n"
        print(f"{sl_no + 1}:\n{suggestion}\n{'-' * 100}\n")
