from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)


def hybrid_chunk(doc):
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=224,
    )
    tokenizer.tokenizer.model_max_length = 10**9
    chunker = HybridChunker(
        tokenizer=tokenizer,
        repeat_table_header=True,
        merge_peers=True,
        omit_header_on_overflow=False,
        always_emit_headings=False,
    )
    chunks = list(chunker.chunk(doc))

    return chunks
