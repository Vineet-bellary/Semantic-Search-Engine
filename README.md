# Semantic Search Engine

A lightweight semantic search system that retrieves relevant information from a collection of documents using both traditional lexical retrieval and dense vector representations.

The project was built from scratch to understand and engineer the complete retrieval pipeline:

```
Documents
    |
    v
Text Extraction
    |
    v
Chunking
    |
    v
Text Representation
    |
    v
Vector Index
    |
    v
Query Retrieval
    |
    v
Ranked Results
```

---

## Features Implemented

### Multi-document ingestion

- Supports ingestion of multiple PDF documents from a data directory.
- Automatically discovers PDF files.
- Extracts document metadata:
  - Document name
  - Page number
  - Text content

Example chunk schema:

```json
{
  "chunk_id": "document_name_page_chunk",
  "document_name": "example.pdf",
  "page_number": 3,
  "text_chunk": "content extracted from document"
}
```

---

## Version 0: TF-IDF Retrieval System

### Overview

Implemented a classical information retrieval pipeline using TF-IDF representations.

Pipeline:

```
PDF Documents

      |
      v

Text Extraction

      |
      v

Fixed-size Chunking

      |
      v

TF-IDF Vectorization

      |
      v

Cosine Similarity Search

      |
      v

Top-K Ranked Results
```

---

### Components

#### Text Extraction

Implemented PDF parsing pipeline using `pypdf`.

Extracted:

- Page-level text
- Document metadata
- Page references

---

#### Chunking

Implemented configurable fixed-size chunking.

Parameters:

```python
chunk_size
overlap_size
```

Experimented with different chunk sizes:

| Chunk Size | Accuracy@1 | Accuracy@3 |
| ---------- | ---------: | ---------: |
| 100        |     26.19% |     57.14% |
| 125        |     30.95% |     52.38% |
| 150        |     30.95% |     64.29% |
| 175        |     35.71% |     59.52% |
| 200        |     33.33% |     52.38% |
| 250        |     23.81% |     52.38% |
| 300        |     21.43% |     42.86% |

Best TF-IDF configuration:

```
Chunk Size: 175
Accuracy@1: 35.71%
Accuracy@3: 59.52%
```

---

#### Retrieval

Implemented cosine similarity ranking:

```
Query
 |
 v
TF-IDF Vector
 |
 v
Similarity against document chunks
 |
 v
Top-K chunks
```

Returned:

- Document name
- Page number
- Similarity score
- Relevant text chunk

---

## Version 1: Dense Embedding Retrieval

### Motivation

TF-IDF relies on exact keyword overlap.

To support semantic matching:

Example:

```
Query:
"How do neural networks learn?"

Document:
"Deep architectures optimize parameters using gradient descent..."
```

The system was upgraded to use dense embeddings.

---

## Embedding Pipeline

Implemented using a pretrained Sentence Transformer model.

Pipeline:

```
Text Chunk

      |
      v

Sentence Transformer

      |
      v

384-dimensional embedding vector

      |
      v

Vector similarity search
```

Current embedding model:

```
all-MiniLM-L6-v2
```

Embedding output:

```
387 chunks

↓

torch.Size([387, 384])
```

---

## Storage Design

Separated metadata and vector storage.

### chunks.json

Stores document metadata:

```json
{
  "chunk_id": "...",
  "document_name": "...",
  "page_number": "...",
  "text_chunk": "..."
}
```

### embeddings.pt

Stores dense vectors:

```
Tensor:
(number_of_chunks, embedding_dimension)
```

Example:

```
torch.Size([387,384])
```

---

## Retrieval Architecture

Current retrieval flow:

```
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

torch.topk()

    |
    v

Relevant Chunk Indices

    |
    v

Document Results
```

Ranking implemented using PyTorch cosine similarity.

---

## Evaluation Framework

Built an automated evaluation pipeline.

Features:

- JSON-based evaluation queries
- Automated retrieval testing
- Accuracy@1 measurement
- Accuracy@3 measurement
- Chunking strategy comparison

Evaluation metrics:

```
Accuracy@1:
Did the first retrieved result contain the expected answer?

Accuracy@3:
Was the answer present in the top three retrieved results?
```

---

## Embedding Retrieval Experiments

Tested different chunking strategies.

| Chunk Configuration | Accuracy@1 | Accuracy@3 |
| ------------------- | ---------: | ---------: |
| 100                 |     11.90% |     38.10% |
| 125                 |     40.48% |     54.76% |
| 150                 |     21.43% |     52.38% |
| 150 + overlap 50    |     14.29% |     54.76% |
| 200                 |     11.90% |     38.10% |
| 250                 |      9.52% |     35.71% |

Best embedding configuration:

```
Chunk Size: 125 words
Overlap: 0

Accuracy@1: 40.48%
Accuracy@3: 54.76%
```

---

## Engineering Practices

Implemented:

- Modular project architecture
- Separate ingestion and retrieval pipelines
- Config-driven parameters
- Persistent ingestion artifacts
- Evaluation framework
- Package-based Python structure
- GPU-aware embedding generation
- Reproducible experiments

---

## Project Structure

```
semantic-search-engine/

src/
|
└── semantic_search_engine/
    |
    ├── ingestion/
    │   ├── document_loader.py
    │   ├── text_extraction.py
    │   ├── chunking.py
    │   └── representation/
    │       ├── vectorization.py
    │       └── embedding.py
    |
    ├── retrieval/
    │   ├── query.py
    │   ├── similarity.py
    │   └── process_query.py
    |
    ├── utils/
    │   └── save_load_metadata.py
    |
    ├── evaluation/
    |
    ├── config.py
    └── main.py
```

---

## Future Improvements

Planned improvements:

- Semantic chunking
- Structure-aware chunking
- Hybrid retrieval:
  - TF-IDF + embeddings
- Vector database integration
- Metadata filtering
- Query expansion
- Reranking models
- multiple file formate acceptance
- Retrieval-Augmented Generation (RAG)

---

## Key Learnings

Through this project:

- Designed an end-to-end information retrieval system.
- Compared lexical and semantic retrieval approaches.
- Built evaluation-driven optimization workflows.
- Learned how chunking affects retrieval quality.
- Implemented dense vector search without relying on frameworks such as LangChain.
