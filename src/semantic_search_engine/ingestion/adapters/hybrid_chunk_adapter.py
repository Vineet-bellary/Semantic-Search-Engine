from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

from docling_core.transforms.chunker import DocChunk

from semantic_search_engine.models.chunk import ChunkProvenance, SSEChunk


class HybridChunkAdapter:
    """Convert Docling hybrid chunks into the project-level SSEChunk schema."""

    def adapt_chunk(self, chunk: DocChunk, order: int) -> SSEChunk:
        """Adapt a single Docling chunk into an SSEChunk."""

        meta = chunk.meta
        document_name = self._get_document_name(
            meta.origin.filename if meta.origin else None
        )

        headings = list(meta.headings or [])
        labels: list[str] = []
        doc_refs: list[str] = []
        provenance: list[ChunkProvenance] = []
        pages: set[int] = set()

        for item in meta.doc_items:
            label = getattr(item, "label", None)
            if label:
                labels.append(str(label))

            self_ref = getattr(item, "self_ref", None)
            if self_ref:
                doc_refs.append(str(self_ref))

            for prov in getattr(item, "prov", []) or []:
                page_no = getattr(prov, "page_no", None)
                bbox = getattr(prov, "bbox", None)
                charspan = getattr(prov, "charspan", None)

                if page_no is not None:
                    pages.add(page_no)

                if bbox is None or charspan is None:
                    continue

                provenance.append(
                    ChunkProvenance(
                        page_no=page_no if page_no is not None else -1,
                        bbox=(bbox.l, bbox.t, bbox.r, bbox.b),
                        coord_origin=str(getattr(bbox, "coord_origin", "")),
                        charspan=(charspan[0], charspan[1]),
                    )
                )

        # Docling marks meta.captions as deprecated; keep the SSE field for
        # schema compatibility and populate it only when non-deprecated sources
        # are available.
        captions: list[str] = []

        return SSEChunk(
            chunk_id=f"{Path(document_name).stem}_{order:06d}",
            document_name=document_name,
            order=order,
            content=chunk.text,
            headings=headings,
            pages=sorted(pages),
            labels=labels,
            doc_refs=doc_refs,
            provenance=provenance,
            captions=captions,
        )

    def extract_info_chunk(self, chunks: Iterable[DocChunk]) -> list[SSEChunk]:
        """Adapt an iterable of Docling chunks into SSEChunk objects."""

        return [self.adapt_chunk(chunk, order) for order, chunk in enumerate(chunks)]

    def _get_document_name(self, filename: str | None) -> str:
        if filename:
            return filename
        return "unknown_document"


if __name__ == "__main__":
    from semantic_search_engine.ingestion.chunkers.hybrid_chunker import hybrid_chunk
    from semantic_search_engine.ingestion.parsers.doc_to_doclingobj import (
        configure_converter,
        parse_doc,
    )

    pdf_path = Path(r"D:\SSE\data\docling_documentation.pdf")
    converter = configure_converter()
    doc = parse_doc(pdf_path, converter)
    chunks = hybrid_chunk(doc)
    adapter = HybridChunkAdapter()
    adapted = adapter.extract_info_chunk(chunks)
    print(f"Adapted {len(adapted)} chunks from {pdf_path.name}")
    print(f"First chunk: {adapted[0] if adapted else 'No chunks available'}")
