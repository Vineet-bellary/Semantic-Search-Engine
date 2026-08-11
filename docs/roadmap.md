# Roadmap

## Completed Milestones

### V1 — Baseline Markdown Pipeline

- Manual PDF-to-Markdown conversion.
- Section-based chunking driven by heading detection (`retrieval_chunker.py`, `markdown_parser.py`).
- TF-IDF vectorization for query and chunk similarity.
- Basic top-k retrieval.

### V2 — Sentence-Transformer Embeddings

- Replaced TF-IDF with dense embeddings using `all-MiniLM-L6-v2`.
- Introduced cosine similarity ranking via PyTorch.
- Added persistence: `chunks.json` and `embeddings.pt`.

### V3 — Evaluation Framework

- Added `eval.py` with Accuracy@1 and Accuracy@3 metrics.
- Introduced `tests/evaluation_queries.json` with heading-path targets.
- Normalization utilities for document names and heading text.
- Per-document and per-target accuracy breakdowns.
- Miss reporting with top-k predicted matches.

### V4 — Docling Hybrid Chunk Pipeline _(current)_

- Integrated Docling PDF conversion with `DOCLING_LAYOUT_EGRET_LARGE`, OCR, and table-structure extraction.
- Replaced Markdown-based chunking with Docling `HybridChunker`.
- Introduced `SSEChunk` and `ChunkProvenance` as the unified internal schema.
- `HybridChunkAdapter` normalizes raw `DocChunk` metadata into `SSEChunk`.
- Ingestion, search, and evaluation all operate on the same schema.
- Expanded evaluation query set to cover newly added PDFs.
- CUDA acceleration support throughout the pipeline.

---

## Near-Term (V5)

- **Relaxed heading matching** — add a configurable soft-match mode to handle verbose financial/legal headings without breaking strict-match tests.
- **Typed load utility** — return `list[SSEChunk]` directly from `load_ingested_data` as an optional typed mode to avoid repeated `model_validate` calls in search and eval.
- **Adapter regression tests** — unit tests that assert `SSEChunk` field mapping from known `DocChunk` fixtures (headings, pages, labels, provenance).
- **Caption enrichment** — re-evaluate Docling API to populate `SSEChunk.captions` from figure/table caption sources once a non-deprecated path is available.
- **Startup diagnostics** — check model cache, HF token, and CUDA availability at startup and emit actionable warnings rather than silent failures.

---

## Medium-Term

- **Vector database backend** — add an optional path (e.g. FAISS or ChromaDB) alongside the current file-based store. The file-based store stays as the default for portability.
- **Reranking** — add a cross-encoder reranking step between similarity retrieval and result display to improve precision at rank 1.
- **Metadata filtering** — allow search queries to filter by document name, page range, or label type before ranking.
- **Multi-query retrieval** — decompose complex queries into sub-queries and merge result sets.
- **Batch search CLI** — non-interactive mode that reads queries from a file and writes results to JSON, useful for regression testing retrieval quality over time.

---

## Long-Term

- **Retrieval-Augmented Generation (RAG)** — connect retrieved chunks to a local or API-backed LLM to produce grounded answers with source citations.
- **Incremental ingestion** — detect and re-ingest only changed or newly added PDFs without reprocessing the full dataset.
- **Web UI** — lightweight local frontend for query input and result browsing, replacing the current CLI.
- **Pluggable embedding models** — swap `all-MiniLM-L6-v2` for larger or domain-specific models (e.g. `bge-large-en`, `e5-mistral`) via config without code changes.
- **Table and figure retrieval** — separate retrieval paths optimized for tabular content and image-derived text so that figure/table chunks rank appropriately for structured queries.
