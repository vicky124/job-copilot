"""Run the whole pipeline offline: fake LLM, hashing embeddings, in-memory vectors,
fixture jobs, auto-approve. No keys, no network, no Docker.

    python -m scripts.demo_run
"""

from __future__ import annotations

import json
from pathlib import Path

from copilot.adapters.jobsources import FixtureJobSource
from copilot.adapters.llm.offline import offline_llm
from copilot.config.settings import load_settings
from copilot.container import build_container
from copilot.domain.models import CandidateProfile, InterviewQuestion
from copilot.graph import AutoApproveGate, LocalPipelineRunner, initial_state
from copilot.services.query_builder import SearchPreferences

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scripts" / "sample_data"


def main() -> None:
    settings = load_settings(
        ROOT,
        env={
            "APP_ENV": "dev",
            "LLM_PROVIDER_PROFILE": "fake",
            "EMBEDDING_PROFILE": "hashing",
            "VECTOR_STORE": "memory",
            "APP_OUTPUT_DIR": "outputs/demo",
        },
        dotenv=False,
    )
    container = build_container(
        settings,
        llm=offline_llm(),
        job_sources=[FixtureJobSource(DATA / "jobs.json")],
    )
    questions = [InterviewQuestion.model_validate(q) for q in json.loads((DATA / "questions.json").read_text())]
    container.services.indexing.index_questions(questions)

    profile = CandidateProfile.model_validate(json.loads((DATA / "profile.json").read_text()))
    state = initial_state(
        "demo-user",
        SearchPreferences(roles=["backend engineer"], locations=["Bengaluru", "Pune"]),
        profile=profile,
    )
    runner = LocalPipelineRunner(
        container.nodes(AutoApproveGate()), settings.app.pipeline, container.tracer
    )
    result = runner.run(state)

    print("path      :", " -> ".join(result.path))
    for score in result.state.get("scored", []):
        print(f"score     : {score.score:3d}  {score.job_id}  gaps={score.gaps}")
    for job_id, path in result.state.get("artifacts", {}).items():
        print(f"resume    : {job_id} -> {path}")
    for job_id, pack in result.state.get("interview_packs", {}).items():
        print(f"prep      : {job_id}: {len(pack.questions)} questions")
    print("errors    :", result.state.get("errors") or "none")


if __name__ == "__main__":
    main()
