"""Greenhouse public job board: GET boards-api.greenhouse.io/v1/boards/{slug}/jobs."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from copilot.adapters.jobsources.enrich import enrich_posting, strip_html
from copilot.config.settings import CompanyEntry
from copilot.domain.india import is_india_location
from copilot.domain.matching import matches_query
from copilot.domain.models import CompanyType, JobPosting, JobQuery
from copilot.ports.http import HttpClient, HttpError

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"


class GreenhouseJobSource:
    def __init__(
        self,
        http: HttpClient,
        companies: list[CompanyEntry],
        location_filter: list[str],
        endpoint: str = DEFAULT_ENDPOINT,
    ) -> None:
        self._http = http
        self._companies = companies
        self._filter = location_filter
        self._endpoint = endpoint

    @property
    def name(self) -> str:
        return "greenhouse"

    def search(self, query: JobQuery) -> list[JobPosting]:
        results: list[JobPosting] = []
        for company in self._companies:
            try:
                data = self._http.get_json(self._endpoint.format(slug=company.slug))
            except HttpError as exc:
                log.warning("greenhouse board %s failed: %s", company.slug, exc)
                continue
            for raw in data.get("jobs", []) if isinstance(data, dict) else []:
                posting = self._to_posting(raw, company)
                if posting and matches_query(posting, query):
                    results.append(posting)
        return results[: query.limit]

    def _to_posting(self, raw: dict[str, Any], company: CompanyEntry) -> JobPosting | None:
        location = str((raw.get("location") or {}).get("name", ""))
        if not is_india_location(location, self._filter):
            return None
        return enrich_posting(
            JobPosting(
                source=self.name,
                external_id=f"{company.slug}-{raw.get('id')}",
                title=str(raw.get("title", "")).strip(),
                company=company.name,
                location=location,
                url=str(raw.get("absolute_url", "")),
                description=strip_html(str(raw.get("content", ""))),
                posted_at=_parse_dt(raw.get("first_published") or raw.get("updated_at")),
                company_type=CompanyType(company.segment),
            )
        )


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
