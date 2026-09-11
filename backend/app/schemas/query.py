from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Override the number of chunks to retrieve"
    )
    diverse_sources: bool = Field(
        default=False, description="If true, return at most one chunk per source file"
    )


class SourceItem(BaseModel):
    source: str
    page: int
    chunk: int
    distance: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    embedding_backend: str


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    embedding_backend: str
    ollama_available: bool
