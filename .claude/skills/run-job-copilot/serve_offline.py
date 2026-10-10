"""Serve the FastAPI app fully offline with fixture jobs, so /runs completes.

The fake LLM profile (copilot/adapters/llm/offline.py) already has canned replies;
this adds what env vars can't: the fixture job source (no network to Greenhouse/
Lever) and the sample interview-question bank.

    .venv/Scripts/python .claude/skills/run-job-copilot/serve_offline.py [port]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from copilot.adapters.jobsources import FixtureJobSource  # noqa: E402
from copilot.api.app import create_app  # noqa: E402
from copilot.config.settings import load_settings  # noqa: E402
from copilot.container import build_container  # noqa: E402
from copilot.domain.models import InterviewQuestion  # noqa: E402
from scripts.demo_run import DATA  # noqa: E402


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    settings = load_settings(
        ROOT,
        env={
            "APP_ENV": "dev",
            "LLM_PROVIDER_PROFILE": "fake",
            "EMBEDDING_PROFILE": "hashing",
            "VECTOR_STORE": "memory",
            "APP_OUTPUT_DIR": "outputs/offline",
        },
        dotenv=False,
    )
    container = build_container(settings, job_sources=[FixtureJobSource(DATA / "jobs.json")])
    questions = json.loads((DATA / "questions.json").read_text())
    container.services.indexing.index_questions([InterviewQuestion.model_validate(q) for q in questions])
    uvicorn.run(create_app(settings, container), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
