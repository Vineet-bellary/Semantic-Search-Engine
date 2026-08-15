import json
import torch
from pathlib import Path

from semantic_search_engine.models.chunk import SSEChunk


def save_ingested_data(
    chunks: list[dict] | list[SSEChunk], embeddings: torch.Tensor, output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.json"

    serializable_chunks: list[dict] = []
    for chunk in chunks:
        if isinstance(chunk, SSEChunk):
            serializable_chunks.append(chunk.model_dump(mode="json"))
        else:
            serializable_chunks.append(chunk)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, indent=4, ensure_ascii=False)

    embeddings_path = output_dir / "embeddings.pt"
    torch.save(embeddings.cpu(), embeddings_path)

    print(f"Successfully saved all ingestion assets to {output_dir}")


def load_ingested_data(ingested_data_dir: Path, device: str = "cpu"):
    chunks_path = ingested_data_dir / "chunks.json"
    embeddings_path = ingested_data_dir / "embeddings.pt"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = torch.load(embeddings_path, map_location="cpu")
    embeddings = embeddings.to(device)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Chunk count and embedding count mismatch")
    else:
        print(f"Successfully loaded all ingestion assets from {ingested_data_dir}")

    return chunks, embeddings
