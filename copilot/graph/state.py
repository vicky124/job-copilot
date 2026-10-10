"""Graph state. Every node returns a partial update; dict fields merge per job_id."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from copilot.domain.models import (
    CandidateProfile,
    ClaimViolation,
    EvidencePair,
    FitScore,
    InterviewPack,
    JobPosting,
    JobQuery,
    ReviewDecision,
    TailoredResume,
)
from copilot.services.query_builder import SearchPreferences


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer: per-job dicts are merged, so parallel per-job work never overwrites."""
    return {**(left or {}), **(right or {})}


class CopilotState(TypedDict, total=False):
    # inputs
    user_id: str
    resume_text: str
    preferences: SearchPreferences
    prompt_variant: str
    # parse / search
    profile: CandidateProfile
    query: JobQuery
    query_loops: int
    jobs: list[JobPosting]
    # score / select
    evidence: Annotated[dict[str, list[EvidencePair]], merge_dicts]
    scored: list[FitScore]
    selected_job_ids: list[str]
    # tailor / verify / review
    drafts: Annotated[dict[str, TailoredResume], merge_dicts]
    violations: Annotated[dict[str, list[ClaimViolation]], merge_dicts]
    flags: Annotated[dict[str, list[ClaimViolation]], merge_dicts]  # on user-edited drafts
    tailor_attempts: Annotated[dict[str, int], merge_dicts]
    pending_review: list[str]
    review: Annotated[dict[str, ReviewDecision], merge_dicts]
    # outputs
    artifacts: Annotated[dict[str, str], merge_dicts]
    interview_packs: Annotated[dict[str, InterviewPack], merge_dicts]
    errors: Annotated[list[str], operator.add]


def initial_state(
    user_id: str,
    preferences: SearchPreferences,
    *,
    resume_text: str | None = None,
    profile: CandidateProfile | None = None,
) -> CopilotState:
    if not resume_text and profile is None:
        raise ValueError("provide resume_text or a parsed profile")
    state: CopilotState = {
        "user_id": user_id,
        "preferences": preferences,
        "query_loops": 0,
        "errors": [],
    }
    if resume_text:
        state["resume_text"] = resume_text
    if profile is not None:
        state["profile"] = profile
    return state


# Keys whose updates merge instead of replace (mirrors the Annotated reducers above).
MERGED_KEYS = {
    "evidence",
    "drafts",
    "violations",
    "flags",
    "tailor_attempts",
    "review",
    "artifacts",
    "interview_packs",
}
APPENDED_KEYS = {"errors"}


def apply_update(state: CopilotState, update: dict[str, Any]) -> CopilotState:
    """Apply a node's partial update the same way LangGraph's reducers do."""
    new: dict[str, Any] = dict(state)
    for key, value in update.items():
        if key in MERGED_KEYS:
            new[key] = merge_dicts(new.get(key), value)
        elif key in APPENDED_KEYS:
            new[key] = [*(new.get(key) or []), *value]
        else:
            new[key] = value
    return new  # type: ignore[return-value]
