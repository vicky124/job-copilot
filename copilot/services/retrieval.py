"""Retrieval patterns from the architecture doc: requirement <-> evidence pairs."""

from __future__ import annotations

from copilot.domain.models import Chunk, EvidencePair
from copilot.ports.embeddings import Embedder
from copilot.ports.tracing import Tracer
from copilot.ports.vector_store import VectorStore
from copilot.services.indexing import CollectionNames


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        names: CollectionNames,
        tracer: Tracer,
        min_score: float = 0.25,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._names = names
        self._tracer = tracer
        self._min_score = min_score

    def job_requirements(self, job_id: str, limit: int = 30) -> list[Chunk]:
        """All requirement chunks of one job. Falls back to any section if none are tagged."""
        probe = self._embedder.embed_query("requirements qualifications skills experience")
        for section in ("requirements", None):
            filters: dict[str, str | int | float | bool] = {"job_id": job_id}
            if section:
                filters["section"] = section
            hits = self._store.search(
                self._names.job_descriptions, probe, limit=limit, filters=filters
            )
            if hits:
                return [h.chunk for h in hits]
        return []

    def evidence_for(self, user_id: str, requirements: list[Chunk]) -> list[EvidencePair]:
        pairs: list[EvidencePair] = []
        with self._tracer.span("retrieval.evidence", user_id=user_id, n=len(requirements)) as span:
            misses = 0
            for req in requirements:
                hits = self._store.search(
                    self._names.candidate_profile,
                    self._embedder.embed_query(req.text),
                    limit=1,
                    filters={"user_id": user_id},
                )
                best = hits[0] if hits and hits[0].score >= self._min_score else None
                misses += best is None
                pairs.append(
                    EvidencePair(
                        requirement=req.text,
                        evidence=best.chunk.text if best else None,
                        evidence_chunk_id=best.chunk.id if best else None,
                        similarity=round(best.score, 4) if best else 0.0,
                    )
                )
            span.set_attribute("retrieval_misses", misses)
        return pairs

    def interview_questions(self, query_text: str, limit: int = 15) -> list[Chunk]:
        hits = self._store.search(
            self._names.interview_questions, self._embedder.embed_query(query_text), limit=limit
        )
        return [h.chunk for h in hits]
