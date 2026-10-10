"""In-memory vector store with cosine similarity. Dev mode and tests only."""

from __future__ import annotations

import math
from collections import defaultdict

from copilot.domain.models import Chunk
from copilot.ports.vector_store import Filter, SearchHit


class CollectionNotFoundError(KeyError):
    pass


class DimensionMismatchError(ValueError):
    pass


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._dims: dict[str, int] = {}
        self._points: dict[str, dict[str, tuple[Chunk, list[float]]]] = defaultdict(dict)

    def ensure_collection(self, name: str, dimensions: int) -> None:
        existing = self._dims.get(name)
        if existing is not None and existing != dimensions:
            raise DimensionMismatchError(
                f"collection '{name}' has {existing} dims, asked for {dimensions}"
            )
        self._dims[name] = dimensions

    def _require(self, name: str) -> int:
        if name not in self._dims:
            raise CollectionNotFoundError(name)
        return self._dims[name]

    def upsert(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        dims = self._require(collection)
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != dims:
                raise DimensionMismatchError(f"vector has {len(vector)} dims, expected {dims}")
            self._points[collection][chunk.id] = (chunk, vector)

    def delete_group(self, collection: str, key: str, value: str) -> None:
        self._require(collection)
        points = self._points[collection]
        for point_id in [pid for pid, (c, _) in points.items() if c.metadata.get(key) == value]:
            del points[point_id]

    def search(
        self,
        collection: str,
        vector: list[float],
        *,
        limit: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        self._require(collection)
        hits = [
            SearchHit(chunk=chunk, score=_cosine(vector, vec))
            for chunk, vec in self._points[collection].values()
            if not filters or all(chunk.metadata.get(k) == v for k, v in filters.items())
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def count(self, collection: str) -> int:
        return len(self._points.get(collection, {}))
