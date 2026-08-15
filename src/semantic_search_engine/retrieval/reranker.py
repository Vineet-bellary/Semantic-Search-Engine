from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        pairs = []

        for candidate in candidates:
            heading_text = " > ".join(candidate.get("headings", []))
            content = candidate.get("content", "")
            passage = f"{heading_text}\n{content}" if heading_text else content
            pairs.append((query, passage))

        scores = self.model.predict(pairs)

        rerank_scores = [float(score) for score in scores]
        rerank_min = min(rerank_scores)
        rerank_range = max(rerank_scores) - rerank_min
        retrieval_scores = [
            float(candidate.get("retrieval_score", 0.0)) for candidate in candidates
        ]
        retrieval_min = min(retrieval_scores)
        retrieval_range = max(retrieval_scores) - retrieval_min

        reranked = []
        for candidate, rerank_score, retrieval_score in zip(
            candidates, rerank_scores, retrieval_scores
        ):
            normalized_rerank = (
                (rerank_score - rerank_min) / rerank_range if rerank_range else 0.0
            )
            normalized_retrieval = (
                (retrieval_score - retrieval_min) / retrieval_range
                if retrieval_range
                else 0.0
            )
            combined_score = 0.65 * normalized_rerank + 0.35 * normalized_retrieval
            reranked.append(
                {
                    **candidate,
                    "rerank_score": rerank_score,
                    "combined_score": combined_score,
                }
            )

        return sorted(
            reranked,
            key=lambda candidate: candidate["combined_score"],
            reverse=True,
        )
