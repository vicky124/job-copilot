from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from copilot.domain.models import Chunk

Filter = dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    score: float


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collection(self, name: str, dimensions: int) -> None: ...

    def upsert(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def delete_group(self, collection: str, key: str, value: str) -> None:
        """Delete every point whose metadata[key] == value (e.g. all chunks of a job)."""
        ...

    def search(
        self,
        collection: str,
        vector: list[float],
        *,
        limit: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]: ...
