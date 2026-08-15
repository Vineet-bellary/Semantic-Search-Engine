# Milestone V4.1: Retrieval Optimization

## Status

Completed on 2026-08-15.

## Objective

Improve retrieval precision within the existing artifact-based V4 pipeline without introducing a vector database, changing the embedding model, or changing the `SSEChunk` contract.

## Implemented Changes

### Tokenizer-aligned chunking

- Configured Docling `HybridChunker` with the `sentence-transformers/all-MiniLM-L6-v2` tokenizer.
- Set the chunk budget to 224 tokens, leaving headroom below MiniLM's 256-token limit.
- Kept table-header repetition and peer merging enabled.
- Adjusted tokenizer metadata so Docling can measure oversized source items without emitting misleading transformer sequence-length warnings during its pre-splitting pass.

### Heading-aware embeddings

Chunk embeddings now use:

```text
heading 1 > heading 2 > ...
chunk content
```

The persisted `SSEChunk.content` remains the original chunk content. Heading context is added only to the text sent to the embedding model, allowing section identity to influence semantic similarity without changing the stored schema.

### Query normalization

Query preprocessing now trims and normalizes whitespace while preserving punctuation and non-ASCII characters. This avoids deleting potentially meaningful terms before embedding.

### Hybrid ranking

Ranking remains dense-first but now supports an optional lexical tie-breaker:

```text
final score = 0.85 * cosine similarity + 0.15 * lexical overlap
```

Lexical overlap is computed from non-stopword terms shared between the normalized query and the chunk's headings plus content. The lexical component improves precision for section-specific queries while keeping semantic similarity dominant.

### Cross-encoder reranking

Search and evaluation retrieve the top 20 candidates with the hybrid ranker, then score query-candidate pairs with `cross-encoder/ms-marco-MiniLM-L-6-v2`. The final ordering uses a normalized blend:

```text
final score = 0.65 * cross-encoder score + 0.35 * first-stage score
```

The reranker runs at query time and does not require regenerating the stored embeddings.

## Evaluation

The benchmark contains 70 queries from `tests/evaluation_queries.json`.

| Metric     |    V4 baseline |    V4.1 result |
| ---------- | -------------: | -------------: |
| Accuracy@1 | 54/70 (77.14%) | 58/70 (82.86%) |
| Accuracy@3 | 61/70 (87.14%) | 67/70 (95.71%) |

The final ranking configuration preserved the V4.1 Accuracy@1 improvement and raised Accuracy@3 by an additional two points compared with the original V4 baseline.

## Final Artifacts

The final ingestion run processed 14 PDFs and produced:

- 966 chunks in `ingested_data/chunks.json`
- 966 embeddings in `ingested_data/embeddings.pt`
- Embedding shape: `(966, 384)`

The chunk and embedding counts remain aligned and pass the existing persistence integrity check.

## Known Limitation

Docling still reports an occasional native allocation failure for an individual page:

```text
Stage preprocess failed for run 12, pages [10]: std::bad_alloc
```

The conversion run continues, but the affected page may have incomplete OCR or layout data. Resolving this requires a separate converter-memory experiment because disabling OCR, table extraction, parsed-page generation, or GPU acceleration can change retrieval quality.

## Follow-up Work

- Add regression tests for tokenizer-aligned chunk sizes and hybrid ranking.
- Record per-document and per-target benchmark results as a versioned baseline.
- Investigate the remaining Docling page-level allocation failure.
- Tune candidate count and reranker weighting only after the current pipeline is protected by regression tests.
