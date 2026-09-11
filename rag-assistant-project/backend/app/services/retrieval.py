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
	"what", "is", "the", "a", "an", "of", "in", "on", "to", "for", "and",
	"or", "with", "how", "why", "can", "does", "do", "are", "about", "these",
	"those", "this", "that", "document", "documents", "please", "tell", "me",
}
MIN_CHUNK_TEXT_LENGTH = 200
LOW_QUALITY_MARKERS = ("glossary ", "table of contents", "self-check: answer")
OVERVIEW_QUERY_PATTERNS = (
	r"\bmain (topic|subject|idea|objective)\b",
	r"\b(topic|subject|theme) of (these|the) documents\b",
	r"\b(summarize|summary|overview|general idea|key points)\b",
)


def _terms(text: str) -> set[str]:
	return {
		token for token in re.findall(r"[a-z0-9]+", text.lower())
		if len(token) > 2 and token not in STOP_WORDS
	}


def _low_quality(text: str) -> bool:
	normalized = text.lower().strip()
	return any(marker in normalized[:500] for marker in LOW_QUALITY_MARKERS)


def _is_overview_query(query: str) -> bool:
	normalized = re.sub(r"\s+", " ", query.lower().strip())
	return any(re.search(pattern, normalized) for pattern in OVERVIEW_QUERY_PATTERNS)


class RetrievalService:
	def __init__(self) -> None:
		self.pdf_files = sorted(settings.data_dir.glob("*.pdf"))
		self.chunks: list[dict[str, Any]] = []
		self.chunk_embeddings: list[list[float]] = []
		self.embedding_backend = "none"
		self._embed_texts_fn = None
		self._tfidf_vectorizer = None
		self._tfidf_is_fitted = False
		self.cache_path = settings.vector_dir / "retrieval_cache.pkl"

		self._load_embedding_backend()
		if not self._load_cache():
			self._build_index()
			self._save_cache()

	def _load_embedding_backend(self) -> None:
		try:
			from sentence_transformers import SentenceTransformer

			model = SentenceTransformer(settings.embedding_model)
			self._embed_texts_fn = lambda texts: model.encode(
				texts, normalize_embeddings=True
			).tolist()
			self.embedding_backend = "sentence-transformers"
			return
		except Exception as error:
			logger.warning("Sentence-transformers unavailable: %s", error)

		try:
			from sklearn.feature_extraction.text import TfidfVectorizer

			self._tfidf_vectorizer = TfidfVectorizer(
				max_features=4096,
				stop_words="english",
			)

			def embed(texts: list[str]) -> list[list[float]]:
				if not self._tfidf_is_fitted:
					vectors = self._tfidf_vectorizer.fit_transform(texts)
					self._tfidf_is_fitted = True
				else:
					vectors = self._tfidf_vectorizer.transform(texts)
				return vectors.toarray().tolist()

			self._embed_texts_fn = embed
			self.embedding_backend = "tfidf"
			return
		except Exception as error:
			logger.warning("TF-IDF unavailable: %s", error)

		def hash_embed(texts: list[str], dimensions: int = 384) -> list[list[float]]:
			vectors = []
			for text in texts:
				vector = [0.0] * dimensions
				for token in re.findall(r"[a-z0-9]+", text.lower()):
					vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % dimensions] += 1
				norm = sum(value * value for value in vector) ** 0.5 or 1.0
				vectors.append([value / norm for value in vector])
			return vectors

		self._embed_texts_fn = hash_embed
		self.embedding_backend = "hash"

	def _signature(self) -> dict[str, Any]:
		return {
			"backend": self.embedding_backend,
			"chunk_size": settings.chunk_size,
			"chunk_overlap": settings.chunk_overlap,
			"pdfs": [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in self.pdf_files],
		}

	def _load_cache(self) -> bool:
		if not self.cache_path.exists():
			return False
		try:
			with self.cache_path.open("rb") as file:
				cached = pickle.load(file)
			if cached.get("signature") != self._signature():
				return False
			self.chunks = cached["chunks"]
			self.chunk_embeddings = cached["embeddings"]
			if self.embedding_backend == "tfidf" and cached.get("vectorizer") is not None:
				self._tfidf_vectorizer = cached["vectorizer"]
				self._tfidf_is_fitted = True
			return True
		except Exception as error:
			logger.warning("Could not load retrieval cache: %s", error)
			return False

	def _save_cache(self) -> None:
		settings.vector_dir.mkdir(parents=True, exist_ok=True)
		with self.cache_path.open("wb") as file:
			pickle.dump({
				"signature": self._signature(),
				"chunks": self.chunks,
				"embeddings": self.chunk_embeddings,
				"vectorizer": self._tfidf_vectorizer,
			}, file, protocol=pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def _chunk_text(text: str) -> list[str]:
		words = text.split()
		step = max(1, settings.chunk_size - settings.chunk_overlap)
		return [
			" ".join(words[start:start + settings.chunk_size])
			for start in range(0, len(words), step)
			if words[start:start + settings.chunk_size]
		]

	def _build_index(self) -> None:
		if PdfReader is None:
			logger.warning("pypdf is not installed; no documents indexed")
			return
		for pdf_path in self.pdf_files:
			try:
				reader = PdfReader(str(pdf_path))
				for page_number, page in enumerate(reader.pages, start=1):
					text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
					for chunk_number, chunk in enumerate(self._chunk_text(text), start=1):
						self.chunks.append({
							"id": hashlib.sha1(f"{pdf_path.name}:{page_number}:{chunk_number}".encode()).hexdigest()[:16],
							"text": chunk,
							"metadata": {
								"source": pdf_path.name,
								"page": page_number,
								"chunk": chunk_number,
							},
						})
			except Exception as error:
				logger.warning("Could not index %s: %s", pdf_path.name, error)
		if self.chunks:
			self.chunk_embeddings = self._embed_texts_fn([item["text"] for item in self.chunks])
		logger.info("Indexed %d chunks using %s", len(self.chunks), self.embedding_backend)

	def retrieve(self, query: str, top_k: int | None = None, diverse_sources: bool = False) -> list[dict[str, Any]]:
		if not self.chunks:
			return []
		top_k = top_k or settings.top_k
		query_terms = _terms(query)
		query_embedding = self._embed_texts_fn([query])[0]
		scored = []
		for chunk, vector in zip(self.chunks, self.chunk_embeddings):
			score = sum(left * right for left, right in zip(query_embedding, vector))
			lexical_matches = len(query_terms & set(re.findall(r"[a-z0-9]+", chunk["text"].lower())))
			scored.append((1 - score, lexical_matches, chunk))
		scored.sort(key=lambda item: (item[0], -item[1]))
		max_distance = (
			settings.max_distance_sentence_transformers
			if self.embedding_backend == "sentence-transformers"
			else settings.max_distance_fallback
		)
		overview_query = _is_overview_query(query)
		results = []
		seen_sources = set()
		for distance, lexical_matches, chunk in scored:
			if len(chunk["text"]) < MIN_CHUNK_TEXT_LENGTH or _low_quality(chunk["text"]):
				continue
			if not overview_query and distance > max_distance:
				continue
			source = chunk["metadata"]["source"]
			if diverse_sources and source in seen_sources:
				continue
			seen_sources.add(source)
			results.append({"text": chunk["text"], "metadata": chunk["metadata"], "distance": distance})
			if len(results) >= top_k:
				break
		return results


_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
	global _retrieval_service
	if _retrieval_service is None:
		_retrieval_service = RetrievalService()
	return _retrieval_service
