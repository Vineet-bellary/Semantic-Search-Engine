# Milestone V4: Docling Hybrid Chunk Pipeline

## Summary

This milestone completes the migration from the legacy markdown-chunk schema to a Docling-based hybrid chunking pipeline with a stable internal chunk model.

The ingestion, search, and evaluation paths now operate on a unified chunk structure (`SSEChunk`) and persist both chunk metadata and embeddings for repeatable experiments.

## Goals Achieved

1. Integrated Docling conversion with configurable PDF pipeline options.
2. Added hybrid chunking with structure-aware metadata capture.
3. Introduced `HybridChunkAdapter` to normalize raw Docling chunks into `SSEChunk`.
4. Updated ingestion to persist adapted chunks and tensor embeddings.
5. Updated search to read and display the new chunk schema.
6. Updated evaluation pipeline to validate against heading-path targets.
7. Expanded evaluation set with newly added PDFs.

## Current Architecture

### Ingestion Flow

1. Discover PDFs from `data/`.
2. Convert each PDF into DoclingDocument with configured pipeline options.
3. Run Docling `HybridChunker` to create raw chunks.
4. Adapt raw chunks to `SSEChunk` via `HybridChunkAdapter`.
5. Embed `SSEChunk.content` using sentence-transformer embeddings.
6. Save:
   - `ingested_data/chunks.json`
   - `ingested_data/embeddings.pt`

### Search Flow

1. Load `chunks.json` and `embeddings.pt`.
2. Validate user query and embed it.
3. Run similarity ranking.
4. Parse results as `SSEChunk`.
5. Display top results with document, heading path, pages, score, and text preview.

### Evaluation Flow

1. Load evaluation queries from `tests/evaluation_queries.json`.
2. Compute query embeddings and retrieve top-k matches.
3. Compare predicted document and heading path to expected values.
4. Report Accuracy@1 / Accuracy@3 and per-target/per-document performance.

## Data Model: SSEChunk

`SSEChunk` currently stores:

- `chunk_id`
- `document_name`
- `order`
- `content`
- `headings`
- `pages`
- `labels`
- `doc_refs`
- `provenance`
- `captions`

Notes:

- `captions` is preserved in the schema for compatibility, but currently populated as an empty list in the adapter because Docling `meta.captions` is deprecated.

## Converter and Model Configuration (Current)

- Layout model: `DOCLING_LAYOUT_EGRET_LARGE`
- OCR: enabled
- Table structure: enabled
- Code enrichment: disabled
- Formula enrichment: disabled
- Heading hierarchy: enabled
- Parsed page generation: enabled
- Accelerator: CUDA when available, else CPU

## Milestone Outcomes

1. End-to-end ingestion executes successfully on the current multi-PDF dataset.
2. Search uses the same schema produced by ingestion (no legacy field mismatch).
3. Evaluation is aligned with heading-based targets and supports current data.
4. Newly added PDFs are now represented in evaluation query coverage.

## Known Limitations

1. Some long financial/legal headings are verbose and may increase strict-match brittleness.
2. HF Hub unauthenticated downloads may be slower/rate-limited when model artifacts are not cached.
3. Very long token sequence warnings can still appear during some model operations.
4. Adapter currently does not enrich `captions` due to Docling deprecation changes.

## Recommended Next Steps

1. Add optional relaxed heading matching mode for long/verbose headings.
2. Return `list[SSEChunk]` directly from load utility as an optional typed mode.
3. Introduce regression tests for adapter field mapping and schema integrity.
4. Add offline cache checks and startup diagnostics for model/token availability.
5. Begin vector database integration once retrieval quality plateaus on file-based storage.

## Milestone Scope

This document reflects the current V4 state on branch `v4-document-parsing` after integration of hybrid chunking, schema adaptation, and evaluation updates.
