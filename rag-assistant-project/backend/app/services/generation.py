from __future__ import annotations

import logging
import re
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

UNSUPPORTED_ANSWER = "The answer is not supported by the provided documents."
OLLAMA_AVAILABLE = True

try:
	from ollama import Client
except ImportError:
	Client = None
	OLLAMA_AVAILABLE = False

CHITCHAT_PATTERNS = (
	r"^(hi|hello|hey|good morning|good afternoon|good evening)[!. ]*$",
	r"^(thanks|thank you|thx)[!. ]*$",
	r"^how are you[?.! ]*$",
)


def is_chitchat(question: str) -> bool:
	normalized = re.sub(r"\s+", " ", question.strip().lower())
	return any(re.match(pattern, normalized) for pattern in CHITCHAT_PATTERNS)


def handle_chitchat(question: str) -> str:
	normalized = question.strip().lower()
	if normalized.startswith(("thanks", "thank you", "thx")):
		return "You're welcome! How can I help you?"
	if normalized.startswith("how are you"):
		return "I'm doing well! How can I help you?"
	return "Hello! How can I help you?"


def _build_context(sources: list[dict[str, Any]]) -> str:
	return "\n\n".join(
		f"[{item['metadata']['source']}, page {item['metadata']['page']}]\n{item['text']}"
		for item in sources
		if item.get("text", "").strip()
	)


def _fallback_answer(sources: list[dict[str, Any]]) -> str:
	first = sources[0]
	metadata = first["metadata"]
	excerpt = first["text"].strip()
	if len(excerpt) > 600:
		excerpt = excerpt[:600].rsplit(" ", 1)[0] + "..."
	return f"{excerpt}\n\n[{metadata['source']}, page {metadata['page']}]"


def _generate_with_ollama(question: str, context: str) -> str | None:
	if Client is None:
		return None
	try:
		client = Client(host=getattr(settings, "ollama_host", "http://localhost:11434"))
		response = client.chat(
			model=settings.ollama_model,
			messages=[{
				"role": "user",
				"content": (
					"Answer only from the context. If unsupported, say exactly: "
					f"{UNSUPPORTED_ANSWER}\n\nContext:\n{context}\n\nQuestion: {question}"
				),
			}],
		)
		answer = response.get("message", {}).get("content", "").strip()
		return answer or None
	except Exception as error:
		logger.warning("Ollama unavailable: %s", error)
		return None


def answer_question(question: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
	question = question.strip()
	if is_chitchat(question):
		return {"question": question, "route": "chitchat", "answer": handle_chitchat(question), "sources": []}
	if not sources:
		return {"question": question, "route": "off_topic", "answer": UNSUPPORTED_ANSWER, "sources": []}

	context = _build_context(sources)
	answer = _generate_with_ollama(question, context) or _fallback_answer(sources)
	if answer.strip() == UNSUPPORTED_ANSWER:
		return {"question": question, "route": "off_topic", "answer": UNSUPPORTED_ANSWER, "sources": []}
	return {"question": question, "route": "rag", "answer": answer, "sources": sources}
