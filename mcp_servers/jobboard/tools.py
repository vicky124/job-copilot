"""jobboard-mcp tool logic (read-only). Plain functions so they are unit-testable."""

from __future__ import annotations

from typing import Any

from copilot.adapters.jobsources import PastedJobParser
from copilot.domain.models import JobQuery, WorkMode
from copilot.services.job_search import JobSearchService


class JobBoardTools:
    def __init__(self, search: JobSearchService, source_names: list[str]) -> None:
        self._search = search
        self._sources = source_names
        self._pasted = PastedJobParser()

    def search_jobs(
        self,
        roles: list[str],
        locations: list[str] | None = None,
        work_modes: list[str] | None = None,
        experience_years: float | None = None,
        min_ctc_lpa: float | None = None,
        limit: int = 25,
    ) -> dict[str, Any]:
        query = JobQuery(
            roles=roles,
            locations=locations or [],
            work_modes=[WorkMode(m) for m in work_modes or []],
            exp_years=experience_years,
            ctc_min_lpa=min_ctc_lpa,
            limit=limit,
        )
        result = self._search.search(query)
        return {
            "jobs": [p.model_dump(mode="json") for p in result.postings],
            "source_errors": result.errors,
            "duplicates_removed": result.duplicates_removed,
        }

    def parse_job(self, text: str, url: str = "") -> dict[str, Any]:
        return self._pasted.parse(text, url=url).model_dump(mode="json")

    def list_sources(self) -> list[str]:
        return list(self._sources)
