# Architecture

## Overview

The Semantic Search Engine (SSE) is a local PDF retrieval system. It converts PDFs into structured chunks, embeds them using a sentence-transformer model, and ranks chunks against a user query using dense semantic similarity with a lightweight lexical tie-breaker.

The system has three top-level operations: **ingestion**, **search**, and **evaluation**. All three share a common data model (`SSEChunk`) and the same persistence layer (`ingested_data/`).

---

## Pipeline

```text
PDF files in data/
    │
    ▼
Docling converter  (doc_to_doclingobj.py)
    │  └─ layout model: DOCLING_LAYOUT_EGRET_LARGE
    │  └─ OCR, table structure, heading hierarchy enabled
    │
    ▼
Hybrid chunker  (hybrid_chunker.py)
    │  └─ Docling HybridChunker
    │
    ▼
HybridChunkAdapter  (hybrid_chunk_adapter.py)
    │  └─ raw DocChunk → SSEChunk
    │
    ▼
Embedding model  (embedding.py)
    │  └─ all-MiniLM-L6-v2 via sentence-transformers
    │  └─ heading path + chunk content used for embedding
    │
    ▼
Persistence  (save_load_metadata.py)
    │  └─ ingested_data/chunks.json
    │  └─ ingested_data/embeddings.pt
    │
    ▼
Search / Evaluation
    │  └─ 85% cosine similarity + 15% lexical term overlap
```

---

## Module Structure

```markdown
src/semantic_search_engine/
├── config.py # Paths, model names, thresholds, env vars
├── main.py # Entry point: runs ingestion if needed, then search
├── process_documents.py # Ingestion orchestration
├── search.py # Interactive search loop
├── eval.py # Batch evaluation (Accuracy@k)
│
├── models/
│ └── chunk.py # SSEChunk and ChunkProvenance Pydantic models
│
├── ingestion/
│ ├── parsers/
│ │ ├── doc_to_doclingobj.py # Configures DocumentConverter; returns DoclingDocument
│ │ └── doc_to_markdown.py # Legacy: PDF → Markdown export (unused in V4 pipeline)
│ ├── chunkers/
│ │ ├── hybrid_chunker.py # Thin wrapper around Docling HybridChunker
│ │ └── retrieval_chunker.py # Legacy: section-based chunker for Markdown input
│ ├── adapters/
│ │ ├── hybrid_chunk_adapter.py # DocChunk → SSEChunk with full metadata extraction
│ │ └── markdown_parser.py # Legacy: extracts heading-based sections from Markdown
│ └── encoders/
│ ├── embedding.py # EmbeddingModel wrapping SentenceTransformer
│ └── vectorization.py # Legacy: TF-IDF vectorizer (unused in V4 pipeline)
│
├── retrieval/
│ ├── input_handling/
│ │ ├── process_query.py # Query normalization and validation
│ │ └── query.py # CLI input prompt
│ └── similarity.py # Hybrid dense/lexical ranking
│
└── utils/
├── document_loader.py # File discovery from a directory
└── save_load_metadata.py # JSON + .pt serialization/deserialization
```

---

## Data Model

### `SSEChunk`

Defined in `models/chunk.py` using Pydantic. This is the canonical unit across ingestion, storage, search, and evaluation.

| Field           | Type                    | Description                                          |
| --------------- | ----------------------- | ---------------------------------------------------- |
| `chunk_id`      | `str`                   | `<doc_stem>_<order:06d>`                             |
| `document_name` | `str`                   | Source filename                                      |
| `order`         | `int`                   | Sequential position within the document              |
| `content`       | `str`                   | Text content of the chunk                            |
| `headings`      | `list[str]`             | Heading path from document root to this chunk        |
| `pages`         | `list[int]`             | Page numbers this chunk spans                        |
| `labels`        | `list[str]`             | Docling item labels (e.g. `text`, `table`, `figure`) |
| `doc_refs`      | `list[str]`             | Internal `self_ref` identifiers from Docling         |
| `provenance`    | `list[ChunkProvenance]` | Per-item bounding box and char-span data             |
| `captions`      | `list[str]`             | Reserved for figure/table captions (currently empty) |

### `ChunkProvenance`

| Field          | Type                                |
| -------------- | ----------------------------------- |
| `page_no`      | `int`                               |
| `bbox`         | `tuple[float, float, float, float]` |
| `coord_origin` | `str`                               |
| `charspan`     | `tuple[int, int]`                   |

---

## Key Components

### Docling Converter (`doc_to_doclingobj.py`)

Configures a `DocumentConverter` with:

- Layout model: `DOCLING_LAYOUT_EGRET_LARGE`
- OCR and table-structure extraction enabled
- Code and formula enrichment disabled (speed/stability)
- Heading hierarchy extraction enabled
- CUDA acceleration when available

### HybridChunkAdapter (`hybrid_chunk_adapter.py`)

Maps each raw `DocChunk` from Docling into `SSEChunk`. It iterates over `meta.doc_items` to collect labels, `self_ref` identifiers, bounding boxes, and page numbers. The `captions` field is intentionally left empty because `meta.captions` is deprecated in the current Docling version.

### EmbeddingModel (`embedding.py`)

Wraps `SentenceTransformer` (`all-MiniLM-L6-v2`). Accepts either `SSEChunk` objects or legacy dicts. For canonical chunks, the embedding input contains the heading path followed by the chunk content so section context is retained. The HybridChunker reserves token headroom below the model limit. Embeddings are generated as tensors and saved to disk via `torch.save`.

### Similarity Ranking (`similarity.py`)

Computes cosine similarity between a query vector and all chunk vectors using `torch.nn.functional.cosine_similarity`. When query text and chunk metadata are available, it combines the dense score with exact content and heading term overlap using an 85/15 weighting, then returns the top-k scores and indices via `torch.topk`.

### Evaluation (`eval.py`)

Loads queries from `tests/evaluation_queries.json`. Supports two target schemas:

- **Legacy**: `expected_page_number` — match by document and page.
- **V3 / current**: `expected_heading_path` — match by document and heading path.

Heading matching is normalized and tolerant of suffix matches and leaf-heading matches. Reports Accuracy@1, Accuracy@3, per-document and per-target breakdowns, and a miss list.

---

## Configuration (`config.py`)

| Name                   | Value / Source                               |
| ---------------------- | -------------------------------------------- |
| `ROOT_DIR`             | Project root (2 levels above `config.py`)    |
| `DATA_DIR`             | `ROOT_DIR/data`                              |
| `INGESTED_DATA_DIR`    | `ROOT_DIR/ingested_data`                     |
| `TEST_DIR`             | `ROOT_DIR/tests`                             |
| `EMBEDDING_MODEL`      | `"all-MiniLM-L6-v2"`                         |
| `CONFIDENCE_THRESHOLD` | `0.5`                                        |
| `HF_TOKEN`             | `HF_TOKEN` or `HUGGINGFACE_TOKEN_ID` env var |

---

## Persistence Layout

```text
ingested_data/
  chunks.json       # list of SSEChunk dicts
  embeddings.pt     # torch.Tensor of shape (N, 384)
```

Both artifacts are produced together by ingestion and loaded together by search and evaluation. A count mismatch between the two raises a `ValueError` at load time.

---

## Legacy Code

The following modules remain in the repository but are not part of the active V4 pipeline:

- `ingestion/parsers/doc_to_markdown.py` — early Docling-to-Markdown export path
- `ingestion/chunkers/retrieval_chunker.py` — section-based chunker for Markdown input
- `ingestion/adapters/markdown_parser.py` — heading-based section extractor for Markdown
- `ingestion/encoders/vectorization.py` — TF-IDF vectorizer
- `retrieval/process_query.py::vectorize_query` — TF-IDF query transform
