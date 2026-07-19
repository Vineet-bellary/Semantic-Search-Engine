import torch
import torch.nn.functional as F


def rank_chunks(query_vector, chunks_vectors, num_suggestions=3):

    query_vector = query_vector.unsqueeze(0)

    similarities = F.cosine_similarity(query_vector, chunks_vectors)

    scores, top_k_indices = torch.topk(similarities, k=num_suggestions)

    return scores, top_k_indices
