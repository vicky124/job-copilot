"""Hosted OpenAI embeddings (optional profile). Uses LiteLLM's embedding API."""

from __future__ import annotations

from typing import Any

from copilot.config.settings import EmbeddingProfile


class OpenAIEmbedder:
    def __init__(self, profile: EmbeddingProfile, embedding_fn: Any | None = None) -> None:
        self._profile = profile
        self._fn = embedding_fn

    @property
    def dimensions(self) -> int:
        return self._profile.dimensions

    @property
    def model_name(self) -> str:
        return self._profile.model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._fn is None:  # pragma: no cover - optional dependency
            import litellm  # noqa: PLC0415

            self._fn = litellm.embedding
        result = self._fn(model=self._profile.model, input=texts)
        data = result["data"] if isinstance(result, dict) else result.data
        return [list(map(float, item["embedding"])) for item in data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
