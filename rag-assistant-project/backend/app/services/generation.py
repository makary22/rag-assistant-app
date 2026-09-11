from __future__ import annotations

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False
    logger.warning("Install ollama and start the local service to generate answers.")


UNSUPPORTED_ANSWER = "The answer is not supported by the provided documents."

# Only used to decide whether to clear `sources` on the response. We check
# whether the answer STARTS WITH one of these, not whether it merely
# contains the phrase anywhere -- a substring-anywhere check would clear
# sources for legitimate answers that happen to mention a phrase like
# "cannot be determined" mid-sentence as real content.
UNSUPPORTED_PREFIXES = (
    "the answer is not supported by the provided documents",
    "the provided context does not directly support",
    "the provided documents do not directly support",
    "the text does not directly",
    "i'm sorry",
    "i cannot provide",
)

PROMPT_TEMPLATE = """You are a closed-book document question-answering assistant.
Use ONLY the provided context. Do not use outside knowledge or guess.
If the context does not directly support the answer, respond exactly:
{unsupported_answer}
For a supported answer, write exactly 3 short Markdown bullet points:
- **Definition:** Give the direct definition or answer. [filename, page N]
- **How it works:** Explain the main idea or process. [filename, page N]
- **Why it matters:** State the benefit or importance when supported. [filename, page N]
Replace all bracketed instructions with real content. Do not copy the words "one direct definition or answer".
Do not return only a title, abbreviation, or one sentence. Never omit any bullet or citation.

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, sources: list[dict]) -> str:
    context = "\n\n".join(
        f"[{item['metadata']['source']}, page {item['metadata']['page']}] {item['text']}"
        for item in sources
    )
    return PROMPT_TEMPLATE.format(
        unsupported_answer=UNSUPPORTED_ANSWER,
        context=context,
        question=question,
    )


def generate_with_ollama(prompt: str) -> str:
    if not OLLAMA_AVAILABLE or ollama is None:
        return "Ollama is unavailable. Review the retrieved context below."
    try:
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as error:
        logger.error("Ollama request failed: %s", error)
        return f"Ollama request failed: {error}"


def is_refusal(answer: str) -> bool:
    normalized = answer.strip().lower()
    return any(normalized.startswith(prefix) for prefix in UNSUPPORTED_PREFIXES)


def answer_question(question: str, sources: list[dict]) -> dict:
    """
    sources: the list of retrieved chunk dicts from RetrievalService.retrieve()
    Returns {"question": ..., "answer": ..., "sources": ...} where `sources`
    is cleared to [] whenever there's nothing to ground the answer in.
    """
    if not sources:
        return {
            "question": question,
            "answer": UNSUPPORTED_ANSWER,
            "sources": [],
        }

    prompt = build_prompt(question, sources)
    answer = generate_with_ollama(prompt)

    if is_refusal(answer):
        sources = []

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
