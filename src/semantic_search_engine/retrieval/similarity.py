import torch
import torch.nn.functional as F


def rank_chunks(query_vector, chunks_vectors, num_suggestions=3):
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

    scores, top_k_indices = torch.topk(similarities, k=num_suggestions)

    return scores, top_k_indices
