"""Check every company in config/companies.yaml: does the slug resolve, and does it
have India postings right now?

    python -m scripts.validate_companies            # all companies
    python -m scripts.validate_companies --only-unseen
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

from copilot.adapters.http import UrllibHttpClient
from copilot.config.settings import CompaniesConfig, CompanyEntry, load_settings
from copilot.domain.india import is_india_location
from copilot.ports.http import HttpClient, HttpError


@dataclass
class CompanyCheck:
    company: str
    ats: str
    slug_ok: bool
    total: int = 0
    india: int = 0
    error: str = ""


def _locations(ats: str, raw: dict[str, Any]) -> str:
    if ats == "greenhouse":
        return str((raw.get("location") or {}).get("name", ""))
    cats = raw.get("categories") or {}
    return " / ".join(cats.get("allLocations") or [cats.get("location", "")])


def check_company(http: HttpClient, cfg: CompaniesConfig, company: CompanyEntry) -> CompanyCheck:
    url = cfg.adapters[company.ats].endpoint.format(slug=company.slug)
    try:
        data = http.get_json(url)
    except HttpError as exc:
        return CompanyCheck(company.name, company.ats, slug_ok=False, error=str(exc))
    jobs = data.get("jobs", []) if isinstance(data, dict) else data
    india = sum(
        1
        for j in jobs
        if (company.ats == "lever" and j.get("country") == "IN")
        or is_india_location(_locations(company.ats, j), cfg.location_filter)
    )
    return CompanyCheck(company.name, company.ats, slug_ok=True, total=len(jobs), india=india)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-unseen", action="store_true", help="only india_seen: false")
    args = parser.parse_args(argv)
    cfg = load_settings(".").companies
    http = UrllibHttpClient()
    failures = 0
    print(f"{'company':<14}{'ats':<12}{'slug':<6}{'jobs':>6}{'india':>7}")
    for company in cfg.companies:
        if args.only_unseen and company.india_seen:
            continue
        r = check_company(http, cfg, company)
        failures += not r.slug_ok
        print(f"{r.company:<14}{r.ats:<12}{'ok' if r.slug_ok else 'BAD':<6}{r.total:>6}{r.india:>7}  {r.error}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
