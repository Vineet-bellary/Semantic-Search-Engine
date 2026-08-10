# Semantic Search Engine

This project is a personal semantic search system for PDF documents.

In simple terms: it reads documents, understands them in chunks, and finds the most relevant parts when a user asks a question.

## What This Milestone Delivers (V4)

1. PDF ingestion using Docling.
2. Hybrid chunking for structure-aware document segments.
3. A stable internal schema (`SSEChunk`) for all downstream steps.
4. Embedding generation and persistence for fast retrieval.
5. Search output with document name, heading path, page hints, score, and text preview.
6. Automated evaluation with Accuracy@1 and Accuracy@3.

## High-Level Pipeline

```text
PDF files in data/
    |
    v
Docling conversion
    |
    v
Hybrid chunking
    |
    v
Adapter: raw chunk -> SSEChunk
    |
    v
Embeddings (all-MiniLM-L6-v2)
    |
    v
Save artifacts (chunks.json + embeddings.pt)
    |
    v
Search / Evaluation
```

## Why This Exists

This repository is both:

1. A usable local search tool for personal PDFs.
2. An engineering learning project focused on retrieval quality, chunk design, and measurable evaluation.

## Core Data Model

`SSEChunk` currently stores:

1. `chunk_id`
2. `document_name`
3. `order`
4. `content`
5. `headings`
6. `pages`
7. `labels`
8. `doc_refs`
9. `provenance`
10. `captions`

This unified schema is used across ingestion, storage, search, and evaluation.

## Quick Start

### 1) Setup

Create and activate a Python environment, then install project dependencies.

### 2) Configure environment variables

Create `.env` at project root. Important values include:

```env
HUGGINGFACE_TOKEN_ID=your_hf_token
HF_TOKEN=your_hf_token
```

Both names are useful because different components may read different variable names.

### 3) Ingest documents

From `src/` run:

```powershell
python -m semantic_search_engine.process_documents
```

This creates:

1. `ingested_data/chunks.json`
2. `ingested_data/embeddings.pt`

### 4) Run search

From `src/` run:

```powershell
python -m semantic_search_engine.search
```

### 5) Run evaluation

From `src/` run:

```powershell
python -m semantic_search_engine.eval
```

Evaluation queries are in `tests/evaluation_queries.json`.

## Repository Layout (Current)

```text
SSE/
  data/                  # input PDFs
  ingested_data/         # saved chunks + embeddings
  logs/                  # run logs
  docs/                  # project documentation
  src/semantic_search_engine/
    ingestion/
      adapters/
      chunkers/
      encoders/
      parsers/
    models/
    retrieval/
    process_documents.py
    search.py
    eval.py
  tests/
    evaluation_queries.json
```

## Current Configuration Choices

1. Layout model: `DOCLING_LAYOUT_EGRET_LARGE`
2. OCR: enabled
3. Table structure extraction: enabled
4. Code/formula enrichment: disabled (for stability and speed)
5. Accelerator: CUDA when available, otherwise CPU

## Known Limitations

1. Some documents produce very long headings, which can make strict heading-path matching brittle.
2. If Hugging Face auth is missing, model downloads may be slower or rate-limited.
3. Very long sections can trigger tokenizer length warnings in some model calls.

## Next Milestones

1. Add optional vector database backend.
2. Improve heading/path normalization for more robust evaluation.
3. Add retrieval regression tests around adapter mapping and ranking outputs.
4. Add reranking and metadata-based filtering.

## Documentation

For milestone-level technical summary, see:

1. `docs/milestone_v4_hybrid_chunk_pipeline.md`
