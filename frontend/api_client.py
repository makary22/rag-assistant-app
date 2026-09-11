from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 120


class ApiClientError(Exception):
    """Raised when the backend API returns an error or is unreachable."""


def query(question: str, top_k: int | None = None, diverse_sources: bool = False) -> dict:
    """
    Calls POST /api/query on the backend.

    Returns a dict shaped like:
    {
        "question": str,
        "answer": str,
        "sources": [
            {"source": str, "page": int, "chunk": int, "distance": float, "text": str},
            ...
        ],
        "embedding_backend": str,
    }
    """
    payload = {
        "question": question,
        "diverse_sources": diverse_sources,
    }
    if top_k is not None:
        payload["top_k"] = top_k

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as error:
        raise ApiClientError(
            f"Could not reach the backend at {API_BASE_URL}. "
            f"Is `uvicorn app.main:app` running? ({error})"
        ) from error
    except requests.exceptions.Timeout as error:
        raise ApiClientError(
            f"The backend took longer than {REQUEST_TIMEOUT_SECONDS}s to respond."
        ) from error
    except requests.exceptions.HTTPError as error:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except Exception:
            detail = response.text
        raise ApiClientError(f"Backend returned an error: {detail or error}") from error


def health() -> dict:
    """
    Calls GET /api/health on the backend.

    Returns a dict shaped like:
    {
        "status": str,
        "chunks_indexed": int,
        "embedding_backend": str,
        "ollama_available": bool,
    }
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        raise ApiClientError(f"Could not reach the backend at {API_BASE_URL}. ({error})") from error
