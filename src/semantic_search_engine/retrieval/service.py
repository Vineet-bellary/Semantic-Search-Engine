from __future__ import annotations

from dataclasses import dataclass

import torch

from semantic_search_engine.config import INGESTED_DATA_DIR
from semantic_search_engine.ingestion.encoders.embedding import EmbeddingModel
from semantic_search_engine.models.chunk import SSEChunk
from semantic_search_engine.retrieval.input_handling.process_query import (
    preprocess_query,
    validate_query,
)
from semantic_search_engine.retrieval.reranker import CrossEncoderReranker
from semantic_search_engine.retrieval.similarity import rank_chunks
from semantic_search_engine.utils.save_load_metadata import load_ingested_data


@dataclass(frozen=True)
class SearchResult:
    chunk: SSEChunk
    retrieval_score: float
    rerank_score: float


class SearchService:
    """Load retrieval assets and execute the complete two-stage search path."""

    def __init__(self) -> None:
        self.embedding_model = EmbeddingModel()
        self.reranker = CrossEncoderReranker()
        self.chunks, self.embeddings = load_ingested_data(
            INGESTED_DATA_DIR,
            device=self.embedding_model.device,
        )

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        """Return the top reranked results for a natural-language query."""
        validate_query(query)
        preprocessed_query = preprocess_query(query)
        query_vector = self.embedding_model.embed_query(preprocessed_query)

        candidate_count = min(20, len(self.chunks))
        scores, indices = rank_chunks(
            query_vector,
            self.embeddings,
            num_suggestions=candidate_count,
            query_text=preprocessed_query,
            chunks=self.chunks,
        )

        candidates = []
        for score, index in zip(scores, indices):
            source_index = int(index.item())
            candidate = dict(self.chunks[source_index])
            candidate["source_index"] = source_index
            candidate["retrieval_score"] = float(score.item())
            candidates.append(candidate)

        reranked = self.reranker.rerank(preprocessed_query, candidates)[:limit]
        return [
            SearchResult(
                chunk=SSEChunk.model_validate(candidate),
                retrieval_score=float(candidate["retrieval_score"]),
                rerank_score=float(candidate["rerank_score"]),
            )
            for candidate in reranked
        ]
