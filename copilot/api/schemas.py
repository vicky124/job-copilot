"""HTTP request/response models (kept separate from domain models)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from copilot.api.runs import RunSnapshot
from copilot.domain.models import ApplicationStatus, CandidateProfile, ReviewDecision
from copilot.services.query_builder import SearchPreferences


class StartRunRequest(BaseModel):
    resume_text: str | None = Field(default=None, min_length=50)
    profile: CandidateProfile | None = None
    preferences: SearchPreferences = Field(default_factory=SearchPreferences)


class ReviewRequest(BaseModel):
    decisions: list[ReviewDecision] = Field(min_length=1)


class StatusPatch(BaseModel):
    status: ApplicationStatus


class PastedJobRequest(BaseModel):
    text: str = Field(min_length=50)
    url: str = ""


class RunResponse(BaseModel):
    thread_id: str
    status: str
    path: list[str]
    review: dict[str, Any] | None = None
    scored: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_snapshot(cls, snap: RunSnapshot) -> RunResponse:
        state = snap.state
        return cls(
            thread_id=snap.thread_id,
            status=snap.status,
            path=snap.path,
            review=snap.review_payload,
            scored=[s.model_dump() for s in state.get("scored", [])],
            artifacts=dict(state.get("artifacts", {})),
            errors=list(state.get("errors", [])),
            error=snap.error,
        )
