from pathlib import Path
import torch

from semantic_search_engine.utils.save_load_metadata import load_ingested_data
from semantic_search_engine.ingestion.representation.embedding import EmbeddingModel
from semantic_search_engine.retrieval.process_query import (
    preprocess_query,
    validate_query,
)
from semantic_search_engine.retrieval.query import get_query
from semantic_search_engine.retrieval.similarity import rank_chunks
from semantic_search_engine.config import INGESTED_DATA_DIR, CONFIDENCE_THRESHOLD

embedding_model = EmbeddingModel()


def prepare_query(query: str) -> list[float]:
    preprocessed_query = preprocess_query(query)

    query_vector = embedding_model.embed_query(preprocessed_query)

    return query_vector


def search():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    chunks, embeddings = load_ingested_data(INGESTED_DATA_DIR, device=device)

    if validate_query(query := get_query()):
        query_vector = prepare_query(query)

    scores, top_k_indices = rank_chunks(query_vector, embeddings, num_suggestions=3)

    print(f"\n{'-' * 100}\nRelevant data found from your documents:\n{'-' * 100}\n")

    for sl_no, (score, idx) in enumerate(zip(scores, top_k_indices)):
        # chunk =  chunk_id, document_name, page_number, chunk_text
        if score < CONFIDENCE_THRESHOLD:
            continue

        chunk = chunks[idx]

        suggestion = (
            f"Document Name: {chunk['document_name']}\n"
            f"Page Number: {chunk['page_number']}\n"
            f"Score: {score.item():.2f}\n"
            f"\nChunk Text:\n{chunk['text_chunk']}\n"
        )

        print(f"{'-' * 15} : {sl_no + 1} : {'-' * 15}\n{suggestion}\n{'-' * 100}\n")
