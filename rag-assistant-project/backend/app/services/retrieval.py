from __future__ import annotations

import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "on", "to", "for", "and", "or",
    "with", "how", "why", "can", "does", "do", "are", "about", "give", "explain",
    "using", "these", "books", "answer",
}

MIN_CHUNK_TEXT_LENGTH = 200
LOW_QUALITY_MARKERS = (
    "glossary ",
    "table of contents",
    "see answer",
    "self-check: answer",
    "learning objective:",
)


def _retrieval_terms(text: str) -> set[str]:
    return {
        term for term in re.findall(r"[a-z0-9]+", text.lower())
        if term not in STOP_WORDS and len(term) > 2
    }


def _is_low_quality_chunk(text: str) -> bool:
    normalized = text.lower().strip()
    return any(marker in normalized[:500] for marker in LOW_QUALITY_MARKERS)


class RetrievalService:
    """
    Loads every PDF in settings.data_dir, extracts + cleans + chunks the
    text, builds embeddings (sentence-transformers -> TF-IDF -> hash
    fallback, in that order of preference), indexes them in Chroma when
    available, and exposes retrieve() for the RAG pipeline.

    This mirrors the exploratory logic from the training notebook, but runs
    once at process startup instead of once per notebook cell.
    """

    def __init__(self) -> None:
        self.pdf_files: list[Path] = sorted(settings.data_dir.glob("*.pdf"))
        logger.info("PDFs found: %d", len(self.pdf_files))

        self.extraction_failures: list[dict[str, Any]] = []
        self.page_documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.chunk_embeddings: list[list[float]] = []
        self.embedding_backend: str = "none"

        self._embed_texts_fn = None
        self._tfidf_vectorizer = None
        self._tfidf_is_fitted = False

        self.chroma_client = None
        self.collection = None
        self.collection_name: str | None = None
        self.cache_path = settings.vector_dir / "retrieval_cache.pkl"

        self._load_embedding_backend()
        if not self._load_cache():
            self._extract_pages()
            self._clean_pages()
            self._build_chunks()
            self._attach_metadata()
            self._embed_all_chunks()
            self._save_cache()
        self._init_chroma()

    def _cache_signature(self) -> dict[str, Any]:
        return {
            "embedding_backend": self.embedding_backend,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "pdfs": [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "modified_ns": path.stat().st_mtime_ns,
                }
                for path in self.pdf_files
            ],
        }

    def _load_cache(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            with self.cache_path.open("rb") as cache_file:
                cached = pickle.load(cache_file)
            if cached.get("signature") != self._cache_signature():
                logger.info("Retrieval cache is stale; rebuilding index.")
                return False
            self.page_documents = cached["page_documents"]
            self.extraction_failures = cached["extraction_failures"]
            self.chunks = cached["chunks"]
            self.chunk_embeddings = cached["chunk_embeddings"]
            if self.embedding_backend == "tfidf" and cached.get("tfidf_vectorizer") is not None:
                self._tfidf_vectorizer = cached["tfidf_vectorizer"]
                self._tfidf_is_fitted = True
            logger.info("Loaded retrieval cache: %d chunks", len(self.chunks))
            return True
        except Exception as error:
            logger.warning("Could not load retrieval cache (%s); rebuilding index.", error)
            return False

    def _save_cache(self) -> None:
        settings.vector_dir.mkdir(parents=True, exist_ok=True)
        cached = {
            "signature": self._cache_signature(),
            "page_documents": self.page_documents,
            "extraction_failures": self.extraction_failures,
            "chunks": self.chunks,
            "chunk_embeddings": self.chunk_embeddings,
            "tfidf_vectorizer": self._tfidf_vectorizer,
        }
        temporary_path = self.cache_path.with_suffix(".tmp")
        with temporary_path.open("wb") as cache_file:
            pickle.dump(cached, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(self.cache_path)
        logger.info("Saved retrieval cache: %d chunks", len(self.chunks))

    # ------------------------------------------------------------------ #
    # 1. Extraction
    # ------------------------------------------------------------------ #

    def _extract_pages(self) -> None:
        if PdfReader is None:
            logger.warning(
                "pypdf is not installed; no text will be extracted. "
                "Run: pip install pypdf"
            )
            return

        for pdf_path in self.pdf_files:
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, start=1):
                self.page_documents.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": self._extract_page_text(page, pdf_path.name, page_number),
                })

        logger.info("Extracted %d pages", len(self.page_documents))
        logger.info(
            "Pages skipped because of extraction errors: %d",
            len(self.extraction_failures),
        )

    def _extract_page_text(self, page: Any, source: str, page_number: int) -> str:
        try:
            return page.extract_text() or ""
        except Exception as error:
            self.extraction_failures.append({
                "source": source,
                "page": page_number,
                "error": f"{type(error).__name__}: {error}",
            })
            return ""

    # ------------------------------------------------------------------ #
    # 2. Cleaning
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"-\s*\n\s*", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _clean_pages(self) -> None:
        for item in self.page_documents:
            item["text"] = self._clean_text(item["text"])
        self.page_documents = [item for item in self.page_documents if item["text"]]
        logger.info("Non-empty pages after cleaning: %d", len(self.page_documents))

    # ------------------------------------------------------------------ #
    # 3. Chunking
    # ------------------------------------------------------------------ #

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        words = text.split()
        step = max(1, chunk_size - overlap)
        return [
            " ".join(words[start:start + chunk_size])
            for start in range(0, len(words), step)
            if words[start:start + chunk_size]
        ]

    def _build_chunks(self) -> None:
        for page in self.page_documents:
            pieces = self._chunk_text(page["text"], settings.chunk_size, settings.chunk_overlap)
            for chunk_number, chunk in enumerate(pieces, start=1):
                self.chunks.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk": chunk_number,
                })
        logger.info("Created %d chunks", len(self.chunks))

    def _attach_metadata(self) -> None:
        for item in self.chunks:
            item["id"] = hashlib.sha1(
                f"{item['source']}:{item['page']}:{item['chunk']}".encode()
            ).hexdigest()[:16]
            item["metadata"] = {
                "source": item["source"],
                "page": item["page"],
                "chunk": item["chunk"],
            }

    # ------------------------------------------------------------------ #
    # 4. Embedding backend selection (sentence-transformers -> TF-IDF -> hash)
    # ------------------------------------------------------------------ #

    def _load_embedding_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(settings.embedding_model)

            def embed_texts(texts: list[str]) -> list[list[float]]:
                return model.encode(texts, normalize_embeddings=True).tolist()

            self._embed_texts_fn = embed_texts
            self.embedding_backend = "sentence-transformers"
            logger.info("Loaded embedding model: %s", settings.embedding_model)
            return
        except Exception as sentence_error:
            logger.warning(
                "SentenceTransformers unavailable (%s); trying TF-IDF.", sentence_error
            )

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf_vectorizer = TfidfVectorizer(max_features=4096, stop_words="english")

            def embed_texts(texts: list[str]) -> list[list[float]]:
                if not self._tfidf_is_fitted:
                    vectors = self._tfidf_vectorizer.fit_transform(texts)
                    self._tfidf_is_fitted = True
                else:
                    vectors = self._tfidf_vectorizer.transform(texts)
                return vectors.toarray().tolist()

            self._embed_texts_fn = embed_texts
            self.embedding_backend = "tfidf"
            logger.info("Using TF-IDF embeddings.")
            return
        except Exception as tfidf_error:
            logger.warning(
                "TF-IDF unavailable (%s); falling back to hash embeddings.", tfidf_error
            )

        def embed_texts(texts: list[str], dimensions: int = 384) -> list[list[float]]:
            vectors = []
            for text in texts:
                vector = [0.0] * dimensions
                for token in re.findall(r"[a-z0-9]+", text.lower()):
                    vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % dimensions] += 1.0
                norm = sum(value * value for value in vector) ** 0.5 or 1.0
                vectors.append([value / norm for value in vector])
            return vectors

        self._embed_texts_fn = embed_texts
        self.embedding_backend = "hash"
        logger.info("Using hash-based fallback embeddings.")

    def _embed_all_chunks(self) -> None:
        if not self.chunks:
            self.chunk_embeddings = []
            return
        self.chunk_embeddings = self._embed_texts_fn([item["text"] for item in self.chunks])
        logger.info("Embedding backend: %s", self.embedding_backend)
        logger.info("Embedding vectors: %d", len(self.chunk_embeddings))

    # ------------------------------------------------------------------ #
    # 5. Chroma indexing
    # ------------------------------------------------------------------ #

    def _init_chroma(self) -> None:
        try:
            import chromadb

            self.chroma_client = chromadb.PersistentClient(path=str(settings.vector_dir))
            self.collection_name = f"{settings.chroma_collection_prefix}_{self.embedding_backend}"
            self.collection = self.chroma_client.get_or_create_collection(self.collection_name)

            if self.collection.count() != len(self.chunks):
                batch_size = settings.chroma_batch_size
                for start in range(0, len(self.chunks), batch_size):
                    end = start + batch_size
                    batch = self.chunks[start:end]
                    self.collection.upsert(
                        ids=[item["id"] for item in batch],
                        documents=[item["text"] for item in batch],
                        metadatas=[item["metadata"] for item in batch],
                        embeddings=self.chunk_embeddings[start:end],
                    )
            else:
                logger.info("Chroma collection already contains %d chunks; skipping upsert.", len(self.chunks))
            logger.info(
                "Chroma collection '%s' contains %d chunks",
                self.collection_name,
                self.collection.count(),
            )
        except Exception as error:
            self.chroma_client = None
            self.collection = None
            logger.warning(
                "Chroma unavailable (%s); retrieval will use in-memory cosine search.", error
            )

    # ------------------------------------------------------------------ #
    # 6. Retrieval
    # ------------------------------------------------------------------ #

    def _max_distance(self) -> float:
        if self.embedding_backend == "sentence-transformers":
            return settings.max_distance_sentence_transformers
        return settings.max_distance_fallback

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        diverse_sources: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.chunks:
            return []

        top_k = top_k or settings.top_k
        max_distance = self._max_distance()

        query_terms = _retrieval_terms(query)
        query_tokens = re.findall(r"[A-Za-z0-9]+", query)
        required_terms = {token.lower() for token in query_tokens[1:] if token[:1].isupper()}
        minimum_term_matches = 2 if len(query_terms) >= 3 else 1

        query_embedding = self._embed_texts_fn([query])[0]

        # Retrieve a wider candidate pool before lexical/page-quality filtering.
        # The closest few vectors can be covers, indexes, or glossary pages.
        candidate_k = max(top_k * 20, top_k)
        if diverse_sources:
            candidate_k = max(candidate_k, top_k * max(len(self.pdf_files), 1))

        # Use the freshly built vectors so the active embedding model and the
        # current process always share the same vector space. Chroma remains
        # persisted above for storage, but stale collections must not shadow
        # the freshly built index after model/backend changes.
        scores = [
            sum(a * b for a, b in zip(query_embedding, vector))
            for vector in self.chunk_embeddings
        ]
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:candidate_k]
        results = [
            {
                "text": self.chunks[index]["text"],
                "metadata": self.chunks[index]["metadata"],
                "distance": 1 - scores[index],
            }
            for index in ranked
        ]

        results = [
            result for result in results
            if len(result["text"]) >= MIN_CHUNK_TEXT_LENGTH
            and not _is_low_quality_chunk(result["text"])
            and result["distance"] <= max_distance
            and len(query_terms & _retrieval_terms(result["text"])) >= minimum_term_matches
            and required_terms <= _retrieval_terms(result["text"])
        ]

        results.sort(
            key=lambda result: (
                result["distance"],
                -sum(result["text"].lower().count(term) for term in query_terms),
            )
        )

        if diverse_sources:
            unique_results = []
            seen_sources = set()
            for result in results:
                source = result["metadata"]["source"]
                if source not in seen_sources:
                    unique_results.append(result)
                    seen_sources.add(source)
                if len(unique_results) == top_k:
                    break
            return unique_results

        return results[:top_k]


_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    """
    Singleton accessor. FastAPI will call this via Depends() on every
    request, but the expensive indexing work in __init__ only runs once,
    the first time it's called (e.g. on app startup).
    """
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
