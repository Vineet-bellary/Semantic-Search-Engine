import torch
from sentence_transformers import SentenceTransformer

from semantic_search_engine.config import EMBEDDING_MODEL, HF_TOKEN
from semantic_search_engine.models.chunk import SSEChunk


class EmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = EMBEDDING_MODEL
        self.model = self.load_model()

    def load_model(self):
        model = SentenceTransformer(
            self.model_name, device=self.device, token=HF_TOKEN, local_files_only=True
        )
        return model

    def embed_chunks(self, chunks: list[dict] | list[SSEChunk]):
        """Embed chunk content from either legacy dicts or SSEChunk objects."""

        texts = []
        for chunk in chunks:
            if isinstance(chunk, SSEChunk):
                texts.append(chunk.content)
            else:
                texts.append(chunk.get("content", chunk.get("text_chunk", "")))

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
