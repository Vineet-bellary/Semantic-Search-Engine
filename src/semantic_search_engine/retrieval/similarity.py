from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def rank_chunks(query_vector, chunks_vectors, num_suggestions=3):
    similarities = cosine_similarity(query_vector, chunks_vectors)

    scores = similarities.flatten()
    top_k_indices = scores.argsort()[::-1][:num_suggestions]

    return scores, top_k_indices
