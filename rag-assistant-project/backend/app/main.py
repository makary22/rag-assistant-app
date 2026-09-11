from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import query
from app.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Assistant",
    description="Retrieval-augmented question answering over a local PDF library.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api", tags=["query"])


@app.on_event("startup")
def warm_up_retrieval_index() -> None:
    # Building the index (PDF extraction + chunking + embeddings + Chroma)
    # is expensive, so we do it once here instead of on the first request.
    from app.services.retrieval import get_retrieval_service

    logger.info("Warming up retrieval index on startup...")
    service = get_retrieval_service()
    logger.info(
        "Retrieval index ready: %d chunks, backend=%s",
        len(service.chunks),
        service.embedding_backend,
    )


@app.get("/")
def root() -> dict:
    return {"service": "RAG Document Assistant", "status": "running"}
