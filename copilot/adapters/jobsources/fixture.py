"""Job source backed by a local JSON file. Offline dev and demos."""

from __future__ import annotations

import json
from pathlib import Path

from copilot.adapters.jobsources.enrich import enrich_posting
from copilot.domain.matching import matches_query
from copilot.domain.models import JobPosting, JobQuery


class FixtureJobSource:
    def __init__(self, path: Path, name: str = "fixture") -> None:
        self._path = path
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def search(self, query: JobQuery) -> list[JobPosting]:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        postings = [enrich_posting(JobPosting.model_validate(item)) for item in raw]
        return [p for p in postings if matches_query(p, query)][: query.limit]
