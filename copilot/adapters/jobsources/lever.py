"""Lever public postings: GET api.lever.co/v0/postings/{slug}?mode=json."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from copilot.adapters.jobsources.enrich import enrich_posting, strip_html
from copilot.config.settings import CompanyEntry
from copilot.domain.india import is_india_location
from copilot.domain.matching import matches_query
from copilot.domain.models import CompanyType, JobPosting, JobQuery, WorkMode
from copilot.ports.http import HttpClient, HttpError

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.lever.co/v0/postings/{slug}?mode=json"

_WORKPLACE = {
    "onsite": WorkMode.ONSITE,
    "hybrid": WorkMode.HYBRID,
    "remote": WorkMode.REMOTE_INDIA,
}


class LeverJobSource:
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
        return "lever"

    def search(self, query: JobQuery) -> list[JobPosting]:
        results: list[JobPosting] = []
        for company in self._companies:
            try:
                data = self._http.get_json(self._endpoint.format(slug=company.slug))
            except HttpError as exc:
                log.warning("lever board %s failed: %s", company.slug, exc)
                continue
            for raw in data if isinstance(data, list) else []:
                posting = self._to_posting(raw, company)
                if posting and matches_query(posting, query):
                    results.append(posting)
        return results[: query.limit]

    def _to_posting(self, raw: dict[str, Any], company: CompanyEntry) -> JobPosting | None:
        categories = raw.get("categories") or {}
        all_locations = categories.get("allLocations") or [categories.get("location", "")]
        location = " / ".join(str(loc) for loc in all_locations if loc)
        if raw.get("country") != "IN" and not is_india_location(location, self._filter):
            return None
        lists = "\n".join(
            f"{item.get('text', '')}\n{strip_html(str(item.get('content', '')))}"
            for item in raw.get("lists") or []
        )
        description = "\n".join(
            part
            for part in (
                raw.get("descriptionPlain") or strip_html(str(raw.get("description", ""))),
                lists,
                raw.get("additionalPlain", ""),
            )
            if part
        )
        created = raw.get("createdAt")
        return enrich_posting(
            JobPosting(
                source=self.name,
                external_id=f"{company.slug}-{raw.get('id')}",
                title=str(raw.get("text", "")).strip(),
                company=company.name,
                location=location,
                url=str(raw.get("hostedUrl", "")),
                description=description,
                posted_at=datetime.fromtimestamp(created / 1000, tz=UTC) if created else None,
                work_mode=_WORKPLACE.get(str(raw.get("workplaceType", "")).lower(), WorkMode.UNKNOWN),
                company_type=CompanyType(company.segment),
            )
        )
