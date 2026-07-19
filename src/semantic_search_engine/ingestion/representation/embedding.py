import os
import torch
from sentence_transformers import SentenceTransformer

from semantic_search_engine.config import EMBEDDING_MODEL, HUGGINGFACE_TOKEN_ID

os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN_ID


class EmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = EMBEDDING_MODEL
        self.model = self.load_model()

    def load_model(self):
        model = SentenceTransformer(self.model_name, device=self.device)
        return model

    def embed_chunks(self, chunks: list[dict]):
        texts = [chunk["text_chunk"] for chunk in chunks]

        embeddings = self.model.encode(
            texts, convert_to_tensor=True, show_progress_bar=True, device=self.device
        )

        return embeddings

    def embed_query(self, query: str):
        embedd_query = self.model.encode(
            query,
            convert_to_tensor=True,
        )

        return embedd_query
