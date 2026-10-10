"""Test doubles and builders shared by the unit tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from copilot.adapters.llm import FakeLLMClient
from copilot.config.settings import Settings, load_settings
from copilot.domain.models import CandidateProfile, JobPosting, JobQuery, WorkExperience
from copilot.ports.http import HttpError

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "scripts" / "sample_data"

OFFLINE_ENV = {
    "APP_ENV": "dev",
    "LLM_PROVIDER_PROFILE": "fake",
    "EMBEDDING_PROFILE": "hashing",
    "VECTOR_STORE": "memory",
}


def offline_settings(output_dir: Path | None = None, **overrides: str) -> Settings:
    env = {**OFFLINE_ENV, **overrides}
    if output_dir is not None:
        env["APP_OUTPUT_DIR"] = str(output_dir)
    settings = load_settings(ROOT, env=env, dotenv=False)
    if output_dir is not None:
        settings = settings.model_copy(update={"output_dir": output_dir})
    return settings


def make_profile(**overrides: Any) -> CandidateProfile:
    data: dict[str, Any] = {
        "full_name": "Asha Rao",
        "email": "asha.rao@example.com",
        "phone": "+91 98765 43210",
        "location": "Bengaluru",
        "summary": "Backend engineer with 4 years building Python services for payments.",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Kubernetes", "AWS"],
        "experience": [
            WorkExperience(
                title="Software Engineer II",
                company="PayQuick",
                start=date(2022, 4, 1),
                bullets=[
                    "Built FastAPI services handling 2 million UPI transactions a day",
                    "Cut p95 latency by 40% by adding Redis caching in front of PostgreSQL",
                    "Moved 12 services to Kubernetes with Docker-based CI/CD",
                ],
                skills=["Python", "FastAPI", "Redis", "PostgreSQL", "Kubernetes", "Docker"],
            ),
            WorkExperience(
                title="Software Engineer",
                company="Billdesk Labs",
                start=date(2020, 7, 1),
                end=date(2022, 3, 31),
                bullets=["Wrote reconciliation jobs in Python on AWS"],
                skills=["Python", "AWS"],
            ),
        ],
        "education": ["B.Tech Computer Science, VTU, 2020"],
        "total_years": 4.2,
        "current_ctc_lpa": 16,
        "expected_ctc_lpa": 24,
        "notice_period_days": 30,
    }
    data.update(overrides)
    return CandidateProfile(**data)


def make_job(**overrides: Any) -> JobPosting:
    data: dict[str, Any] = {
        "source": "test",
        "external_id": "j1",
        "title": "Backend Engineer",
        "company": "Acme Fintech",
        "location": "Bengaluru",
        "locations": ["Bengaluru"],
        "description": (
            "Requirements:\n- 3-5 yrs Python backend\n- FastAPI\n- PostgreSQL and Redis\n"
            "- Kubernetes\nNotice period max 30 days. CTC 18-28 LPA."
        ),
        "required_skills": ["python", "fastapi", "postgresql", "redis", "kubernetes"],
        "exp_min": 3,
        "exp_max": 5,
        "ctc_min_lpa": 18,
        "ctc_max_lpa": 28,
        "notice_period_days": 30,
    }
    data.update(overrides)
    return JobPosting(**data)


def make_query(**overrides: Any) -> JobQuery:
    data: dict[str, Any] = {"roles": ["backend engineer"], "locations": ["Bengaluru"]}
    data.update(overrides)
    return JobQuery(**data)


GOOD_TAILOR = {
    "summary": "Backend engineer with 4 years building Python and FastAPI payment services.",
    "bullets": [
        "Built FastAPI services handling 2 million UPI transactions a day",
        "Cut p95 latency by 40% with Redis caching in front of PostgreSQL",
    ],
    "change_log": ["Led with FastAPI"],
}

FABRICATED_TAILOR = {
    "summary": "Backend engineer with 9 years of Kafka and Go expertise.",
    "bullets": ["Led a Kafka migration that saved 75% of infra cost"],
    "change_log": [],
}


def pipeline_llm(tailor_replies: list[dict[str, Any]] | None = None) -> FakeLLMClient:
    """LLM whose tailor replies come from a queue (to script retries)."""
    replies = list(tailor_replies or [GOOD_TAILOR])

    class _Queued(FakeLLMClient):
        def complete(self, tier, messages, *, json_mode=False, tags=None):  # type: ignore[no-untyped-def]
            prompt = "\n".join(m.content for m in messages)
            if "tailor a resume" in prompt or "ATS optimisation" in prompt:
                text = json.dumps(replies.pop(0) if len(replies) > 1 else replies[0])
                self.rules = [("", text)]
            elif "STAR format" in prompt:
                self.rules = [("", json.dumps({"answers": {}}))]
            else:
                self.rules = []
            return super().complete(tier, messages, json_mode=json_mode, tags=tags)

    return _Queued(default="Good overlap with the core stack.")


@dataclass
class FakeHttpClient:
    """Maps URL prefix -> JSON payload or exception. Records every request."""

    routes: dict[str, Any] = field(default_factory=dict)
    requests: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)

    def get_json(self, url, *, params=None, headers=None, timeout=15.0):  # type: ignore[no-untyped-def]
        self.requests.append((url, params))
        for prefix, payload in self.routes.items():
            if url.startswith(prefix):
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise HttpError(url, 404, "no route")


@dataclass
class StaticJobSource:
    postings: list[JobPosting]
    name: str = "static"
    error: Exception | None = None
    queries: list[JobQuery] = field(default_factory=list)

    def search(self, query: JobQuery) -> list[JobPosting]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return list(self.postings)
