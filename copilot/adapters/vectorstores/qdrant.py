"""Qdrant adapter. Collection names carry the embedding suffix
(e.g. ``job_descriptions_v1_nomic768``) so switching embedding profile never mixes vectors.
"""

from __future__ import annotations

import uuid
from typing import Any

from copilot.domain.models import Chunk
from copilot.ports.vector_store import Filter, SearchHit


class QdrantVectorStore:
    def __init__(self, url: str, api_key: str | None = None, client: Any = None) -> None:
        if client is None:  # pragma: no cover - needs qdrant-client and a server
            from qdrant_client import QdrantClient  # noqa: PLC0415 - optional dependency

            client = QdrantClient(url=url, api_key=api_key)
        self._client = client

    @staticmethod
    def _models() -> Any:  # pragma: no cover - optional dependency
        from qdrant_client import models  # noqa: PLC0415

        return models

    def ensure_collection(self, name: str, dimensions: int) -> None:  # pragma: no cover
        m = self._models()
        if self._client.collection_exists(name):
            info = self._client.get_collection(name)
            size = info.config.params.vectors.size
            if size != dimensions:
                raise ValueError(f"Qdrant collection '{name}' is {size}-dim, expected {dimensions}")
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=m.VectorParams(size=dimensions, distance=m.Distance.COSINE),
        )

    def upsert(  # pragma: no cover
        self, collection: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        m = self._models()
        points = [
            m.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id)),
                vector=vector,
                payload={"chunk_id": chunk.id, "text": chunk.text, **chunk.metadata},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def delete_group(self, collection: str, key: str, value: str) -> None:  # pragma: no cover
        m = self._models()
        selector = m.FilterSelector(
            filter=m.Filter(must=[m.FieldCondition(key=key, match=m.MatchValue(value=value))])
        )
        self._client.delete(collection_name=collection, points_selector=selector)

    def search(  # pragma: no cover
        self,
        collection: str,
        vector: list[float],
        *,
        limit: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        m = self._models()
        qfilter = (
            m.Filter(
                must=[
                    m.FieldCondition(key=k, match=m.MatchValue(value=v)) for k, v in filters.items()
                ]
            )
            if filters
            else None
        )
        result = self._client.query_points(
            collection_name=collection, query=vector, limit=limit, query_filter=qfilter
        )
        hits: list[SearchHit] = []
        for point in result.points:
            payload = dict(point.payload or {})
            chunk = Chunk(
                id=str(payload.pop("chunk_id", point.id)),
                text=str(payload.pop("text", "")),
                metadata=payload,
            )
            hits.append(SearchHit(chunk=chunk, score=float(point.score)))
        return hits
