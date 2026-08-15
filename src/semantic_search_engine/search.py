import torch

from semantic_search_engine.utils.save_load_metadata import load_ingested_data
from semantic_search_engine.ingestion.encoders.embedding import EmbeddingModel
from semantic_search_engine.models.chunk import SSEChunk
from semantic_search_engine.retrieval.input_handling.process_query import (
    preprocess_query,
    validate_query,
)
from semantic_search_engine.retrieval.input_handling.query import get_query
from semantic_search_engine.retrieval.reranker import CrossEncoderReranker
from semantic_search_engine.retrieval.similarity import rank_chunks
from semantic_search_engine.config import INGESTED_DATA_DIR, CONFIDENCE_THRESHOLD

embedding_model = EmbeddingModel()


def prepare_query(query: str) -> list[float]:
    """Preprocess the query and generate its embedding vector using the embedding model.

    Args:
        query (str): The input query string.

    Returns:
        list[float]: The embedding vector of the preprocessed query.
    """
    preprocessed_query = preprocess_query(query)

    query_vector = embedding_model.embed_query(preprocessed_query)

    return query_vector


def search():
    """
    Perform semantic search on the ingested data based on the user's query.
    This function loads the ingested data, validates the user's query, generates the query embedding,
    and ranks the chunks based on their similarity to the query. It then displays the top relevant chunks to the user.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    chunks, embeddings = load_ingested_data(INGESTED_DATA_DIR, device=device)

    query = get_query()
    if not validate_query(query):
        print("Invalid query. Please try again.")
        return

    preprocessed_query = preprocess_query(query)
    query_vector = embedding_model.embed_query(preprocessed_query)

    candidate_count = min(20, len(chunks))
    scores, candidate_indices = rank_chunks(
        query_vector,
        embeddings,
        num_suggestions=candidate_count,
        query_text=preprocessed_query,
        chunks=chunks,
    )

    candidates = []
    for score, idx in zip(scores, candidate_indices):
        candidate = dict(chunks[int(idx)])
        candidate["source_index"] = int(idx)
        candidate["retrieval_score"] = float(score.item())
        candidates.append(candidate)

    reranked_candidates = CrossEncoderReranker().rerank(
        preprocessed_query,
        candidates,
    )[:3]

    print(f"\n{'-' * 100}\nRelevant data found from your documents:\n{'-' * 100}\n")

    for sl_no, candidate in enumerate(reranked_candidates, start=1):
        if candidate["retrieval_score"] < CONFIDENCE_THRESHOLD:
            continue

        chunk = SSEChunk.model_validate(candidate)
        heading_path = " > ".join(chunk.headings) if chunk.headings else "N/A"
        pages_text = (
            ", ".join(str(page) for page in chunk.pages) if chunk.pages else "N/A"
        )

        suggestion = (
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Document Name: {chunk.document_name}\n"
            f"Heading Path: {heading_path}\n"
            f"Pages: {pages_text}\n"
            f"Retrieval Score: {candidate['retrieval_score']:.2f}\n"
            f"Reranker Score: {candidate['rerank_score']:.2f}\n"
            f"\nChunk Text:\n{chunk.content}\n"
        )

        print(f"{'=' * 75} : {sl_no} : {'=' * 75}\n\n{suggestion}\n")
