from docling.chunking import HybridChunker


def hybrid_chunk(doc):

    chunker = HybridChunker()
    chunks = list(chunker.chunk(doc))

    return chunks
