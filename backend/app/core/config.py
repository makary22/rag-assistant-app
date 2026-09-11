from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    # This file lives at backend/app/core/config.py, so parents[3] is the
    # project root (the folder that contains both `data/` and `backend/`).
    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = project_root / "data"
    vector_dir: Path = project_root / "backend" / "data" / "vector_store"

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_model: str = "llama3.2"

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 4
    max_distance_sentence_transformers: float = 0.55
    max_distance_fallback: float = 0.99

    # Chroma
    chroma_batch_size: int = 5000
    chroma_collection_prefix: str = "rag_documents"

    class Config:
        env_file = ".env"


settings = Settings()
