"""Fan out a query to every JobSource, isolate failures, dedupe across sources."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from copilot.domain.models import JobPosting, JobQuery
from copilot.ports.job_source import JobSource
from copilot.ports.tracing import Tracer

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    postings: list[JobPosting]
    errors: dict[str, str] = field(default_factory=dict)
    duplicates_removed: int = 0


def dedupe(postings: list[JobPosting]) -> tuple[list[JobPosting], int]:
    """Drop repeats by job_id, then by content hash (same JD on two boards).
    Keeps the first occurrence, so put preferred sources first."""
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    unique: list[JobPosting] = []
    for posting in postings:
        if posting.job_id in seen_ids or posting.content_hash in seen_hashes:
            continue
        seen_ids.add(posting.job_id)
        seen_hashes.add(posting.content_hash)
        unique.append(posting)
    return unique, len(postings) - len(unique)


class JobSearchService:
    def __init__(self, sources: list[JobSource], tracer: Tracer, max_workers: int = 4) -> None:
        self._sources = sources
        self._tracer = tracer
        self._max_workers = max_workers

    def search(self, query: JobQuery) -> SearchResult:
        errors: dict[str, str] = {}
        collected: list[JobPosting] = []

        def run(source: JobSource) -> tuple[str, list[JobPosting] | Exception]:
            try:
                with self._tracer.span("jobsource.search", source=source.name):
                    return source.name, source.search(query)
            except Exception as exc:  # noqa: BLE001 - one bad source must not fail the run
                return source.name, exc

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for name, outcome in pool.map(run, self._sources):
                if isinstance(outcome, Exception):
                    log.warning("source %s failed: %s", name, outcome)
                    errors[name] = str(outcome)
                else:
                    collected.extend(outcome)

        unique, removed = dedupe(collected)
        return SearchResult(postings=unique[: query.limit], errors=errors, duplicates_removed=removed)
