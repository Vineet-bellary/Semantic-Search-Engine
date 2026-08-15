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
        model = SentenceTransformer(self.model_name, device=self.device, token=HF_TOKEN)
        return model

    def embed_chunks(self, chunks: list[dict] | list[SSEChunk]):
        """Embed chunk content from either legacy dicts or SSEChunk objects."""

        texts = []
        for chunk in chunks:
            if isinstance(chunk, SSEChunk):
                texts.append(self._chunk_embedding_text(chunk))
            else:
                content = chunk.get("content", chunk.get("text_chunk", ""))
                headings = chunk.get("headings", [])
                heading_text = " > ".join(headings)
                texts.append(f"{heading_text}\n{content}" if heading_text else content)

        embeddings = self.model.encode(
            texts, convert_to_tensor=True, show_progress_bar=True, device=self.device
        )

        return embeddings

    def _chunk_embedding_text(self, chunk: SSEChunk) -> str:
        """Include section context so embeddings retain document structure."""
        heading_text = " > ".join(chunk.headings)
        return f"{heading_text}\n{chunk.content}" if heading_text else chunk.content

    def embed_query(self, query: str):
        embedd_query = self.model.encode(
            query,
            convert_to_tensor=True,
        )

        return embedd_query
