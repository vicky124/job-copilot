"""Graph nodes. Each takes the state and returns a partial update.

Nodes hold no business logic of their own: they call services (Single
Responsibility) that were injected through the constructor (Dependency Inversion).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from copilot.config.settings import PipelineConfig
from copilot.domain.models import Decision, FitScore, JobPosting, TailoredResume
from copilot.graph.gates import ReviewGate
from copilot.graph.state import CopilotState
from copilot.ports.documents import DocumentRenderer
from copilot.ports.tracing import Tracer
from copilot.services.applications import ApplicationService
from copilot.services.claim_verifier import ClaimVerifier
from copilot.services.experiments import ABRouter
from copilot.services.fit_scorer import FitScorer
from copilot.services.indexing import IndexingService
from copilot.services.interview_prep import InterviewPrepService
from copilot.services.job_search import JobSearchService
from copilot.services.query_builder import QueryBuilder
from copilot.services.resume_parser import ResumeParser
from copilot.services.retrieval import Retriever
from copilot.services.tailor import TailorService

log = logging.getLogger(__name__)

TAILOR_EXPERIMENT = "tailor_resume"


@dataclass
class PipelineServices:
    """Everything the nodes need. Built by the composition root (copilot.container)."""

    resume_parser: ResumeParser
    query_builder: QueryBuilder
    job_search: JobSearchService
    indexing: IndexingService
    retriever: Retriever
    fit_scorer: FitScorer
    tailor: TailorService
    verifier: ClaimVerifier
    renderer: DocumentRenderer
    interview_prep: InterviewPrepService
    ab_router: ABRouter
    applications: ApplicationService
    tracer: Tracer
    pipeline: PipelineConfig
    output_dir: Path


class PipelineNodes:
    def __init__(self, services: PipelineServices, review_gate: ReviewGate) -> None:
        self.s = services
        self.gate = review_gate

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _jobs_by_id(state: CopilotState) -> dict[str, JobPosting]:
        return {j.job_id: j for j in state.get("jobs", [])}

    @staticmethod
    def _is_final(state: CopilotState, job_id: str) -> bool:
        """Approved or rejected by the user: the model must not touch it again."""
        decision = state.get("review", {}).get(job_id)
        return decision is not None and decision.decision in {Decision.APPROVE, Decision.REJECT}

    def _variant(self, state: CopilotState) -> str:
        return state.get("prompt_variant") or self.s.ab_router.variant(
            TAILOR_EXPERIMENT, state["user_id"]
        )

    # -------------------------------------------------------------------- nodes
    def parse_resume(self, state: CopilotState) -> dict[str, Any]:
        profile = state.get("profile") or self.s.resume_parser.parse_text(state["resume_text"])
        self.s.indexing.index_profile(state["user_id"], profile)
        return {"profile": profile, "prompt_variant": self._variant(state)}

    def build_query(self, state: CopilotState) -> dict[str, Any]:
        if "query" in state:  # looping back: no job cleared the threshold
            return {"query": state["query"].widened(), "query_loops": state.get("query_loops", 0) + 1}
        return {"query": self.s.query_builder.build(state["profile"], state["preferences"])}

    def search_jobs(self, state: CopilotState) -> dict[str, Any]:
        result = self.s.job_search.search(state["query"])
        errors = [f"source {name} failed: {msg}" for name, msg in result.errors.items()]
        return {"jobs": result.postings, "errors": errors}

    def index_jobs(self, state: CopilotState) -> dict[str, Any]:
        for job in state.get("jobs", []):
            self.s.indexing.index_job(job)
        return {}

    def score_fit(self, state: CopilotState) -> dict[str, Any]:
        profile = state["profile"]
        scored: list[FitScore] = []
        evidence = {}
        for job in state.get("jobs", []):
            with self.s.tracer.span("node.score_fit", job_id=job.job_id):
                requirements = self.s.retriever.job_requirements(job.job_id)
                pairs = self.s.retriever.evidence_for(state["user_id"], requirements)
                evidence[job.job_id] = pairs
                scored.append(self.s.fit_scorer.score(profile, job, pairs))
        scored.sort(key=lambda s: s.score, reverse=True)
        return {"scored": scored, "evidence": evidence}

    def select_jobs(self, state: CopilotState) -> dict[str, Any]:
        cfg = self.s.pipeline
        chosen = [s.job_id for s in state.get("scored", []) if s.score >= cfg.fit_threshold]
        return {"selected_job_ids": chosen[: cfg.max_jobs_to_tailor]}

    def tailor_resume(self, state: CopilotState) -> dict[str, Any]:
        jobs = self._jobs_by_id(state)
        drafts, attempts = {}, {}
        prior_attempts = state.get("tailor_attempts", {})
        violations = state.get("violations", {})
        for job_id in state.get("selected_job_ids", []):
            if self._is_final(state, job_id):
                continue
            needs_work = job_id not in state.get("drafts", {}) or violations.get(job_id)
            if not needs_work or prior_attempts.get(job_id, 0) > self.s.pipeline.max_tailor_retries:
                continue
            drafts[job_id] = self.s.tailor.tailor(
                state["profile"],
                jobs[job_id],
                state.get("evidence", {}).get(job_id, []),
                prompt_variant=state.get("prompt_variant", "production"),
                violations=violations.get(job_id),
            )
            attempts[job_id] = prior_attempts.get(job_id, 0) + 1
            self.s.applications.record_draft(state["user_id"], job_id, drafts[job_id])
        return {"drafts": drafts, "tailor_attempts": attempts}

    def verify_claims(self, state: CopilotState) -> dict[str, Any]:
        """Model drafts with violations go back to the tailor. Drafts the user edited
        are never rewritten by the model: their violations become review flags."""
        profile = state["profile"]
        decided = state.get("review", {})
        violations, flags = {}, {}
        for job_id, draft in state.get("drafts", {}).items():
            if job_id not in state.get("selected_job_ids", []) or self._is_final(state, job_id):
                continue
            found = self.s.verifier.verify(profile, draft)
            if job_id in decided and decided[job_id].decision is Decision.EDIT:
                violations[job_id], flags[job_id] = [], found
            else:
                violations[job_id] = found
        return {"violations": violations, "flags": flags}

    def human_review(self, state: CopilotState) -> dict[str, Any]:
        decided = state.get("review", {})
        pending = [
            job_id
            for job_id in state.get("selected_job_ids", [])
            if job_id in state.get("drafts", {})
            and (job_id not in decided or decided[job_id].decision is Decision.EDIT)
        ]
        if not pending:
            return {"pending_review": []}
        payload = {
            "type": "review_drafts",
            "drafts": [
                {
                    "job_id": job_id,
                    "draft": state["drafts"][job_id].model_dump(),
                    "violations": [
                        v.model_dump()
                        for v in (
                            state.get("violations", {}).get(job_id, [])
                            + state.get("flags", {}).get(job_id, [])
                        )
                    ],
                }
                for job_id in pending
            ],
        }
        decisions = self.gate.request(payload)
        drafts: dict[str, TailoredResume] = {}
        for job_id, decision in decisions.items():
            if decision.decision is Decision.EDIT:
                drafts[job_id] = self.s.tailor.apply_edits(state["drafts"][job_id], decision)
        return {"review": decisions, "drafts": drafts, "pending_review": pending}

    def generate_docs(self, state: CopilotState) -> dict[str, Any]:
        artifacts: dict[str, str] = {}
        for job_id, decision in state.get("review", {}).items():
            if decision.decision is not Decision.APPROVE or job_id in state.get("artifacts", {}):
                continue
            doc = self.s.renderer.render_resume(
                state["profile"], state["drafts"][job_id], self.s.output_dir / state["user_id"]
            )
            artifacts[job_id] = str(doc.path)
            self.s.applications.approve(state["user_id"], job_id, str(doc.path))
        return {"artifacts": artifacts}

    def interview_prep(self, state: CopilotState) -> dict[str, Any]:
        jobs = self._jobs_by_id(state)
        packs = {}
        errors: list[str] = []
        for job_id in state.get("artifacts", {}):
            try:
                packs[job_id] = self.s.interview_prep.build_pack(state["profile"], jobs[job_id])
            except Exception as exc:  # noqa: BLE001 - prep is best-effort; docs are already done
                log.warning("interview prep failed for %s: %s", job_id, exc)
                errors.append(f"interview_prep {job_id}: {exc}")
        return {"interview_packs": packs, "errors": errors}

    def as_mapping(self) -> dict[str, Any]:
        """Node name -> callable, for the graph builders."""
        from copilot.graph.spec import GRAPH_SPEC  # noqa: PLC0415 - avoid import cycle

        return {name: getattr(self, name) for name in GRAPH_SPEC.nodes}
