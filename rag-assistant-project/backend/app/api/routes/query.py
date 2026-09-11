from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.query import HealthResponse, QueryRequest, QueryResponse, SourceItem
from app.services.generation import OLLAMA_AVAILABLE, answer_question
from app.services.retrieval import RetrievalService, get_retrieval_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    retrieval: RetrievalService = Depends(get_retrieval_service),
) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    retrieved = retrieval.retrieve(
        request.question,
        top_k=request.top_k,
        diverse_sources=request.diverse_sources,
    )

    result = answer_question(request.question, retrieved)

    sources = [
        SourceItem(
            source=item["metadata"]["source"],
            page=item["metadata"]["page"],
            chunk=item["metadata"]["chunk"],
            distance=round(item["distance"], 4),
            text=item["text"],
        )
        for item in result["sources"]
    ]

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        embedding_backend=retrieval.embedding_backend,
    )


@router.get("/health", response_model=HealthResponse)
def health(
    retrieval: RetrievalService = Depends(get_retrieval_service),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        chunks_indexed=len(retrieval.chunks),
        embedding_backend=retrieval.embedding_backend,
        ollama_available=OLLAMA_AVAILABLE,
    )
