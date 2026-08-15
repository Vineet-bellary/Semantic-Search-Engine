import torch
import torch.nn.functional as F
import re

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "why",
}


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if term not in _STOP_WORDS and len(term) > 1
    }


def _lexical_scores(query_text: str, chunks: list[dict]) -> torch.Tensor:
    query_terms = _terms(query_text)
    if not query_terms:
        return torch.zeros(len(chunks), dtype=torch.float32)

    scores = []
    for chunk in chunks:
        heading_text = " ".join(chunk.get("headings", []))
        chunk_terms = _terms(f"{heading_text} {chunk.get('content', '')}")
        scores.append(len(query_terms & chunk_terms) / len(query_terms))

    return torch.tensor(scores, dtype=torch.float32)


def rank_chunks(
    query_vector,
    chunks_vectors,
    num_suggestions=3,
    query_text=None,
    chunks=None,
):
    """Rank the chunks based on their similarity to the query vector using cosine similarity.

    Args:
        query_vector (torch.Tensor): The embedding vector of the query.
        chunks_vectors (torch.Tensor): The embedding vectors of the chunks.
        num_suggestions (int, optional): The number of top suggestions to return. Defaults to 3.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the scores and indices of the top-k similar chunks.
    """

    query_vector = query_vector.unsqueeze(0)

    similarities = F.cosine_similarity(query_vector, chunks_vectors)

    if query_text is not None and chunks is not None:
        lexical_scores = _lexical_scores(query_text, chunks).to(
            device=similarities.device
        )
        similarities = 0.85 * similarities + 0.15 * lexical_scores

    scores, top_k_indices = torch.topk(similarities, k=num_suggestions)

    return scores, top_k_indices
