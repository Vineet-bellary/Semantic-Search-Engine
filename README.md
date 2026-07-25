# Semantic Search Engine

A lightweight semantic search system built from scratch to understand and engineer a complete retrieval pipeline for personal document search.

The project focuses on document ingestion, retrieval optimization, embedding-based search, and evaluation-driven improvements.

## Overview

### System pipeline

```text
Documents
    |
    v
PDF Conversion
    |
    v
Markdown Structure Extraction
    |
    v
Retrieval-Optimized Chunking
    |
    v
Embedding Generation
    |
    v
Vector Search
    |
    v
Ranked Results
```

## Features Implemented

### Multi-document ingestion

- Supports ingestion of multiple PDF documents.
- Automatically discovers documents from the data directory.
- Converts PDFs into Markdown representation.
- Preserves document structure using heading paths.

### Retrieval-optimized chunks

Current chunk format:

```json
{
  "chunk_id": 0,
  "document_name": "artificial_intelligence_and_machine_learning",
  "heading_path": [
    "Foundations of Artificial Intelligence and Machine Learning",
    "1. Introduction to Machine Learning Fundamentals (Part 1)"
  ],
  "text_chunk": "Machine learning represents a paradigm shift..."
}
```

---

## Version 0: TF-IDF Retrieval System

Implemented a classical information retrieval pipeline.

Pipeline:

```text
PDF Documents
      |
      v
Text Extraction
      |
      v
Fixed-Size Chunking
      |
      v
TF-IDF Vectorization
      |
      v
Cosine Similarity Search
      |
      v
Top-K Results
```

This version was used to understand traditional lexical retrieval.

---

## Version 1: Dense Embedding Retrieval

TF-IDF was limited by keyword overlap.

The system was upgraded to semantic retrieval using dense embeddings.

Embedding pipeline:

```text
Text Chunk
    |
    v
Sentence Transformer
    |
    v
384-dimensional embedding
    |
    v
Similarity Search
```

Current embedding model:

```text
all-MiniLM-L6-v2
```

Similarity ranking uses cosine similarity.

---

## Version 2: Document Structure Experiments

Experimented with structure-aware chunking approaches.

Goals:

- Preserve document hierarchy.
- Improve semantic retrieval.
- Avoid splitting related information.

Experiments included Docling-based chunking strategies.

Observation:

Document structure alone is not enough. Chunks need to be optimized for retrieval rather than only representing document hierarchy.

---

## Version 3: Markdown-Based Retrieval-Optimized Chunking

The current ingestion architecture.

### Motivation

Instead of relying on generic chunkers, the system converts documents into Markdown and builds chunks using:

- Section headings
- Heading hierarchy
- Section content
- Maximum chunk size constraints

This keeps meaningful context together while producing retrieval-friendly chunks.

Pipeline:

```text
PDF
 |
 v
Docling Converter
 |
 v
Markdown Document
 |
 v
Heading Parser
 |
 v
Retrieval Chunker
 |
 v
Embeddings
 |
 v
Vector Index
```

Current chunk statistics:

```text
Total chunks: 165

Minimum chunk size: 254
Maximum chunk size: 327
Average chunk size: 279.12
```

---

## Retrieval System

Query flow:

```text
User Query
    |
    v
Sentence Transformer
    |
    v
Query Embedding
    |
    v
Cosine Similarity
    |
    v
Top-K Chunks
    |
    v
Document + Heading + Content
```

Returned information:

- Document name
- Heading path
- Similarity score
- Relevant text chunk

---

## Evaluation Framework

The project includes an evaluation pipeline.

Metrics:

```text
Accuracy@1:
First retrieved result contains expected information

Accuracy@3:
Expected information appears in top three results
```

Evaluation workflow:

```text
Query
 |
 v
Search System
 |
 v
Top-K Results
 |
 v
Compare With Expected Result
 |
 v
Calculate Retrieval Accuracy
```

---

## Engineering Practices

Implemented:

- Modular Python package architecture
- Separate ingestion and retrieval pipelines
- Config-driven parameters
- Persistent ingestion artifacts
- Evaluation framework
- GPU-accelerated embedding generation
- Reproducible experiments

---

## Project Structure

```text
semantic-search-engine/

src/
|
└── semantic_search_engine/
    |
    ├── ingestion/
    │   ├── document_loader.py
    │   ├── doc_to_markdown.py
    │   ├── markdown_chunker.py
    │   ├── retrieval_chunker.py
    │   └── representation/
    │       ├── embedding.py
    │       └── vectorization.py
    |
    ├── retrieval/
    |
    ├── evaluation/
    |
    ├── config.py
    └── main.py
```

---

## Future Improvements

Planned:

- Hybrid retrieval (BM25 + embeddings)
- Metadata filtering
- Reranking models
- Better evaluation datasets
- Multiple file format support
- Vector database integration
- Query expansion
- Retrieval-Augmented Generation (RAG)

---

## Key Learnings

Through this project:

- Designed an end-to-end semantic retrieval system.
- Compared lexical and semantic retrieval methods.
- Learned how chunking impacts retrieval quality.
- Built evaluation-driven optimization workflows.
- Implemented dense retrieval without depending on frameworks like LangChain.
