from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker, HybridChunker
from pprint import pprint


from semantic_search_engine.config import CHUNK_SIZE, OVERLAP_SIZE

# from semantic_search_engine.ingestion.text_extraction import extract_text_from_pdf
from semantic_search_engine.ingestion.doc_parser import parse_doc

CHUNKER = HybridChunker()

# def chunking(
#     pdf_path: Path, chunk_size: int | None = None, overlap_size: int | None = None
# ) -> list[dict]:
#     chunk_size = chunk_size or CHUNK_SIZE
#     overlap_size = overlap_size or OVERLAP_SIZE
#     pdf_page_data = extract_text_from_pdf(pdf_path)

#     chunks = []
#     for page_data in pdf_page_data:
#         text = page_data["text"]
#         document_name = page_data["document_name"]
#         page_number = page_data["page_number"]

#         words = text.split()
#         step = chunk_size - overlap_size

#         for i in range(0, len(words), step):
#             chunk = words[i : i + chunk_size]
#             text_chunk = " ".join(chunk)
#             chunk_index = len(chunks)
#             document_id = pdf_path.stem.replace(" ", "_").lower()
#             chunk_id = f"{document_id}_{page_number}_{chunk_index + 1}"
#             chunks.append(
#                 {
#                     "chunk_id": chunk_id,
#                     "document_name": document_name,
#                     "page_number": page_number,
#                     "text_chunk": text_chunk,
#                 }
#             )
#     return chunks


def is_valid_chunk(chunk):

    text = chunk.text.strip()

    if len(text.split()) < 20:
        return False

    return True


def chunk_docs(
    pdf_path: Path,
    converter,
    chunk_size: int | None = None,
    overlap_size: int | None = None,
) -> list[dict]:

    chunk_size = chunk_size or CHUNK_SIZE
    overlap_size = overlap_size or OVERLAP_SIZE
    chunker = CHUNKER
    chunks = []

    doc_obj = parse_doc(pdf_path, converter)
    doc_chunks = chunker.chunk(doc_obj)
    doc_chunks = [chunk for chunk in doc_chunks if is_valid_chunk(chunk)]

    for index, chunk in enumerate(doc_chunks):
        chunk_id = f"{pdf_path.stem.replace(' ', '_').lower()}_{index}"
        page_numbers = chunk.meta.doc_items[0].prov[0].page_no
        # page_numbers = sorted(
        #     {item.prov[0].page_no for item in chunk.meta.doc_items if item.prov}
        # )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_name": pdf_path.name,
                "page_number": page_numbers,
                "headings": chunk.meta.headings,
                "text_chunk": chunk.text,
            }
        )
    pprint(f"Created {len(chunks)} chunks for {pdf_path.name}")
    return chunks


# if __name__ == "__main__":
#     converter = DocumentConverter()
#     chunk_docs(
#         Path(r"D:\SSE\data\Artificial Intelligence and Machine Learning.pdf"),
#         converter,
#     )
#     chunk_docs(
#         Path(r"D:\SSE\data\Computer Networks.pdf"),
#         converter,
#     )
#     chunk_docs(
#         Path(r"D:\SSE\data\Economics and Finance.pdf"),
#         converter,
#     )
