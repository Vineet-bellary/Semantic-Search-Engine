from pydantic import BaseModel, Field


class ChunkProvenance(BaseModel):
    page_no: int
    bbox: tuple[float, float, float, float]
    coord_origin: str
    charspan: tuple[int, int]


class SSEChunk(BaseModel):
    chunk_id: str
    document_name: str
    order: int
    content: str
    headings: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    doc_refs: list[str] = Field(default_factory=list)
    provenance: list[ChunkProvenance] = Field(default_factory=list)
    captions: list[str] = Field(default_factory=list)
