"""The pipeline graph, declared once.

Both the LangGraph builder and the local runner are generated from this spec, so
the edges in production and in tests cannot drift apart.

    parse_resume -> build_query -> search_jobs -> index_jobs -> score_fit
    score_fit   --(no job above threshold, loops left)--> build_query (widened)
    score_fit   --> select_jobs -> tailor_resume -> verify_claims
    verify_claims --(violations, retries left)--> tailor_resume
    verify_claims --> human_review
    human_review --(edits)--> verify_claims
    human_review --(approvals)--> generate_docs -> interview_prep -> END
    human_review --(all rejected)--> END
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from copilot.config.settings import PipelineConfig
from copilot.domain.models import Decision
from copilot.graph.state import CopilotState

END = "__end__"
START_NODE = "parse_resume"

Router = Callable[[CopilotState, PipelineConfig], str]


def route_after_score(state: CopilotState, cfg: PipelineConfig) -> str:
    above = [s for s in state.get("scored", []) if s.score >= cfg.fit_threshold]
    if not above and state.get("query_loops", 0) < cfg.max_query_widen_loops:
        return "build_query"
    return "select_jobs"


def route_after_select(state: CopilotState, cfg: PipelineConfig) -> str:
    return "tailor_resume" if state.get("selected_job_ids") else END


def route_after_verify(state: CopilotState, cfg: PipelineConfig) -> str:
    attempts = state.get("tailor_attempts", {})
    decided = state.get("review", {})
    retry = [
        job_id
        for job_id, violations in state.get("violations", {}).items()
        if violations
        and attempts.get(job_id, 0) <= cfg.max_tailor_retries
        and job_id not in decided  # user-touched drafts are never re-tailored
    ]
    return "tailor_resume" if retry else "human_review"


def route_after_review(state: CopilotState, cfg: PipelineConfig) -> str:
    decisions = state.get("review", {})
    pending = state.get("pending_review", [])
    current = [decisions[j] for j in pending if j in decisions]
    if any(d.decision is Decision.EDIT for d in current):
        return "verify_claims"
    if any(d.decision is Decision.APPROVE for d in decisions.values()):
        return "generate_docs"
    return END


@dataclass(frozen=True)
class GraphSpec:
    nodes: tuple[str, ...]
    edges: dict[str, str]
    conditional: dict[str, tuple[Router, tuple[str, ...]]] = field(default_factory=dict)


GRAPH_SPEC = GraphSpec(
    nodes=(
        "parse_resume",
        "build_query",
        "search_jobs",
        "index_jobs",
        "score_fit",
        "select_jobs",
        "tailor_resume",
        "verify_claims",
        "human_review",
        "generate_docs",
        "interview_prep",
    ),
    edges={
        "parse_resume": "build_query",
        "build_query": "search_jobs",
        "search_jobs": "index_jobs",
        "index_jobs": "score_fit",
        "tailor_resume": "verify_claims",
        "generate_docs": "interview_prep",
        "interview_prep": END,
    },
    conditional={
        "score_fit": (route_after_score, ("build_query", "select_jobs")),
        "select_jobs": (route_after_select, ("tailor_resume", END)),
        "verify_claims": (route_after_verify, ("tailor_resume", "human_review")),
        "human_review": (route_after_review, ("verify_claims", "generate_docs", END)),
    },
)
