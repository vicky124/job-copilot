"""Canned replies for provider_profile: fake, so a whole run completes offline.

Without rules FakeLLMClient answers every JSON call with ``{}`` and the pipeline
fails at parse_resume. These rules return a fixed sample candidate (Asha Rao)
whatever resume text is sent; tailored bullets are copied from that profile so
the claim verifier passes. Real parsing needs a real provider.
"""

from __future__ import annotations

import json

from copilot.adapters.llm.fake import FakeLLMClient

SAMPLE_PROFILE = {
    "full_name": "Asha Rao",
    "email": "asha.rao@example.com",
    "phone": "+91 98765 43210",
    "location": "Bengaluru",
    "summary": "Backend engineer with 4 years building Python services for payments.",
    "skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Kubernetes", "AWS"],
    "experience": [
        {
            "title": "Software Engineer II",
            "company": "PayQuick",
            "start": "2022-04-01",
            "end": None,
            "bullets": [
                "Built FastAPI services handling 2 million UPI transactions a day",
                "Cut p95 latency by 40% by adding Redis caching in front of PostgreSQL",
                "Moved 12 services to Kubernetes with Docker-based CI/CD",
            ],
            "skills": ["Python", "FastAPI", "Redis", "PostgreSQL", "Kubernetes", "Docker"],
        },
        {
            "title": "Software Engineer",
            "company": "Billdesk Labs",
            "start": "2020-07-01",
            "end": "2022-03-31",
            "bullets": ["Wrote reconciliation jobs in Python on AWS"],
            "skills": ["Python", "AWS"],
        },
    ],
    "education": ["B.Tech Computer Science, VTU, 2020"],
    "total_years": 4.2,
    "current_ctc_lpa": 16,
    "expected_ctc_lpa": 24,
    "notice_period_days": 30,
}

SAMPLE_TAILORED = {
    "summary": "Backend engineer with 4 years building Python and FastAPI payment services.",
    "bullets": [
        "Built FastAPI services handling 2 million UPI transactions a day",
        "Cut p95 latency by 40% with Redis caching in front of PostgreSQL",
        "Moved 12 services to Kubernetes with Docker-based CI/CD",
    ],
    "change_log": ["Led with FastAPI and PostgreSQL to match the requirements"],
}


def offline_llm() -> FakeLLMClient:
    """Fake client with canned replies keyed on prompt text."""
    tailored = json.dumps(SAMPLE_TAILORED)
    return (
        FakeLLMClient(default="Solid match on the core backend stack.")
        .when("You extract structured data from Indian resumes", json.dumps(SAMPLE_PROFILE))
        .when("You tailor a resume", tailored)
        .when("ATS optimisation expert", tailored)
        .when("STAR format", json.dumps({"answers": {}}))
    )
