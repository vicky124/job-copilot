"""Local embeddings via Ollama (default: nomic-embed-text, 768 dims).

nomic-embed-text expects task prefixes: "search_document: " for indexed text and
"search_query: " for queries. They come from config/llm.yaml.
"""

from __future__ import annotations

from typing import Any

from copilot.config.settings import EmbeddingProfile


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder:
    def __init__(self, profile: EmbeddingProfile, post_json: Any | None = None) -> None:
        """``post_json(url, payload) -> dict``; defaults to an httpx POST."""
        self._profile = profile
        self._post = post_json or _httpx_post
        self._base = (profile.api_base or "http://localhost:11434").rstrip("/")

    @property
    def dimensions(self) -> int:
        return self._profile.dimensions

    @property
    def model_name(self) -> str:
        return self._profile.model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post(f"{self._base}/api/embed", {"model": self._profile.model, "input": texts})
        vectors = data.get("embeddings") if isinstance(data, dict) else None
        if not vectors or len(vectors) != len(texts):
            raise EmbeddingError(f"unexpected Ollama response for {len(texts)} inputs")
        for v in vectors:
            if len(v) != self.dimensions:
                raise EmbeddingError(
                    f"model returned {len(v)} dims but config says {self.dimensions}; "
                    "fix embedding_profiles in config/llm.yaml"
                )
        return [list(map(float, v)) for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self._profile.doc_prefix + t for t in texts]) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._profile.query_prefix + text])[0]


def _httpx_post(url: str, payload: dict[str, Any]) -> Any:  # pragma: no cover - network
    import httpx  # noqa: PLC0415 - optional dependency

    response = httpx.post(url, json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()
