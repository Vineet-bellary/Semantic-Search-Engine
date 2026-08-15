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

### V4.0 — Docling Hybrid Chunk Pipeline

- Integrated Docling PDF conversion with `DOCLING_LAYOUT_EGRET_LARGE`, OCR, and table-structure extraction.
- Replaced Markdown-based chunking with Docling `HybridChunker`.
- Introduced `SSEChunk` and `ChunkProvenance` as the unified internal schema.
- `HybridChunkAdapter` normalizes raw `DocChunk` metadata into `SSEChunk`.
- Ingestion, search, and evaluation all operate on the same schema.
- Expanded evaluation query set to cover newly added PDFs.
- CUDA acceleration support throughout the pipeline.

---

## V4.1 — Chunking & Retrieval Refinement _(completed)_

Improvements within the current V4 architecture — no structural change, refining what V4.0 established.

- **Tokenizer-aligned chunking** — configured `HybridChunker` with the MiniLM tokenizer and a 224-token budget, reserving headroom below the model's 256-token limit.
- **Heading-aware embeddings** — prepend each chunk's heading path to its embedding input while preserving the canonical chunk content separately.
- **Query normalization** — normalize whitespace without deleting punctuation or non-ASCII query content.
- **Hybrid ranking** — combine dense cosine similarity with exact query-term overlap across headings and content using an 85/15 weighting.
- **Cross-encoder reranking** — retrieve 20 candidates with the hybrid ranker, then rerank query-candidate pairs with `cross-encoder/ms-marco-MiniLM-L-6-v2` using a normalized 65/35 blend with first-stage scores.
- **Retrieval quality validation** — regenerated artifacts and evaluated 70 benchmark queries before and after optimization.
- **Relaxed heading matching** — add a configurable soft-match mode to handle verbose financial/legal headings without breaking strict-match tests.
- **Typed load utility** — return `list[SSEChunk]` directly from `load_ingested_data` as an optional typed mode to avoid repeated `model_validate` calls in search and eval.
- **Adapter regression tests** — unit tests that assert `SSEChunk` field mapping from known `DocChunk` fixtures (headings, pages, labels, provenance).
- **Caption enrichment** — re-evaluate Docling API to populate `SSEChunk.captions` from figure/table caption sources once a non-deprecated path is available.
- **Startup diagnostics** — check model cache, HF token, and CUDA availability at startup and emit actionable warnings rather than silent failures.

Measured result for the V4.1 configuration:

- Accuracy@1 improved from 77.14% (54/70) to 82.86% (58/70).
- Accuracy@3 improved from 87.14% (61/70) to 95.71% (67/70).
- Final ingestion produced 966 chunks and embeddings with shape `(966, 384)`.

Known limitation: Docling can still report a native `std::bad_alloc` for an individual PDF page during preprocessing. The ingestion run continues, but that page may have incomplete OCR or layout data.

---

## V5 — Vector Database Backend _(next architectural change)_

- Replace the current local `.pt`/`.json` artifact store with a persistent vector database (e.g. FAISS or ChromaDB), once V4.1's chunking and retrieval quality work is validated.
- This is an architectural change, not an in-place refinement — triggers the version bump from V4.x to V5.
- Reranking — evaluate larger or domain-specific cross-encoders if latency and benchmark coverage justify the additional cost.
- Metadata filtering — allow search queries to filter by document name, page range, or label type before ranking.
- Multi-query retrieval — decompose complex queries into sub-queries and merge result sets.
- Batch search CLI — non-interactive mode that reads queries from a file and writes results to JSON, useful for regression testing retrieval quality over time.

---

## Long-Term

- **Retrieval-Augmented Generation (RAG)** — connect retrieved chunks to a local or API-backed LLM to produce grounded answers with source citations.
- **Incremental ingestion** — detect and re-ingest only changed or newly added PDFs without reprocessing the full dataset.
- **Web UI** — lightweight local frontend for query input and result browsing, replacing the current CLI.
- **Pluggable embedding models** — swap `all-MiniLM-L6-v2` for larger or domain-specific models (e.g. `bge-large-en`, `e5-mistral`) via config without code changes.
- **Table and figure retrieval** — separate retrieval paths optimized for tabular content and image-derived text so that figure/table chunks rank appropriately for structured queries.
- **Support for additional document formats** — extend ingestion beyond PDF (e.g. docx, txt, markdown) once the pipeline architecture is stable.
