"""Adzuna job search API (India: country code ``in``).

Docs: https://developer.adzuna.com/ — terms require "Jobs by Adzuna" attribution
in the UI wherever these results are shown.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from copilot.adapters.jobsources.enrich import enrich_posting
from copilot.domain.models import JobPosting, JobQuery
from copilot.ports.http import HttpClient, HttpError

log = logging.getLogger(__name__)

BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
INR_PER_LAKH = 100_000


class AdzunaJobSource:
    def __init__(
        self,
        http: HttpClient,
        app_id: str,
        app_key: str,
        country: str = "in",
        results_per_page: int = 50,
    ) -> None:
        self._http = http
        self._app_id = app_id
        self._app_key = app_key
        self._country = country
        self._per_page = results_per_page

    @property
    def name(self) -> str:
        return "adzuna"

    def _params(self, query: JobQuery) -> dict[str, str | int]:
        params: dict[str, str | int] = {
            "app_id": self._app_id,
            "app_key": self._app_key,
            "results_per_page": min(self._per_page, query.limit),
            "what_or": " ".join(query.roles),
            "content-type": "application/json",
        }
        if query.keywords:
            params["what"] = " ".join(query.keywords)
        if query.locations:
            params["where"] = query.locations[0]
        if query.ctc_min_lpa:
            params["salary_min"] = int(query.ctc_min_lpa * INR_PER_LAKH)
        return params

    def search(self, query: JobQuery) -> list[JobPosting]:
        url = BASE_URL.format(country=self._country, page=1)
        try:
            data = self._http.get_json(url, params=self._params(query))
        except HttpError as exc:
            log.warning("adzuna search failed: %s", exc)
            return []
        return [self._to_posting(r) for r in (data or {}).get("results", [])][: query.limit]

    def _to_posting(self, raw: dict[str, Any]) -> JobPosting:
        smin, smax = raw.get("salary_min"), raw.get("salary_max")
        created = raw.get("created")
        return enrich_posting(
            JobPosting(
                source=self.name,
                external_id=str(raw.get("id")),
                title=str(raw.get("title", "")).strip(),
                company=str((raw.get("company") or {}).get("display_name", "Unknown")),
                location=str((raw.get("location") or {}).get("display_name", "")),
                url=str(raw.get("redirect_url", "")),
                description=str(raw.get("description", "")),
                posted_at=datetime.fromisoformat(created.replace("Z", "+00:00")) if created else None,
                ctc_min_lpa=round(float(smin) / INR_PER_LAKH, 1) if smin else None,
                ctc_max_lpa=round(float(smax) / INR_PER_LAKH, 1) if smax else None,
            )
        )
