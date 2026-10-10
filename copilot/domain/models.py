"""Domain models. Pure data + invariants; no I/O, no framework imports."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------- Enums
class WorkMode(StrEnum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE_INDIA = "remote-india"
    REMOTE_GLOBAL = "remote-global"
    UNKNOWN = "unknown"


class CompanyType(StrEnum):
    PRODUCT = "product"
    SERVICES = "services"
    GCC = "gcc"
    STARTUP = "startup"
    UNKNOWN = "unknown"


class ApplicationStatus(StrEnum):
    SAVED = "saved"
    TAILORED = "tailored"
    APPROVED = "approved"
    APPLIED = "applied"
    INTERVIEWING = "interviewing"
    OFFER = "offer"
    REJECTED = "rejected"


class Decision(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class RoundType(StrEnum):
    ONLINE_ASSESSMENT = "online_assessment"
    TECHNICAL = "technical"
    SYSTEM_DESIGN = "system_design"
    MANAGERIAL = "managerial"
    HR = "hr"


# ---------------------------------------------------------------- Candidate
class WorkExperience(_Frozen):
    title: str
    company: str
    start: date | None = None
    end: date | None = None  # None means current role
    bullets: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class CandidateProfile(_Frozen):
    """Structured resume. Contact fields are PII and are redacted before LLM calls."""

    full_name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    total_years: float = Field(default=0.0, ge=0)
    current_ctc_lpa: float | None = Field(default=None, ge=0)
    expected_ctc_lpa: float | None = Field(default=None, ge=0)
    notice_period_days: int | None = Field(default=None, ge=0)

    def all_skills(self) -> set[str]:
        """Lower-cased skills from the skills list and every role."""
        skills = {s.lower() for s in self.skills}
        for role in self.experience:
            skills.update(s.lower() for s in role.skills)
        return skills

    def employers(self) -> set[str]:
        return {r.company.lower() for r in self.experience}

    def titles(self) -> set[str]:
        return {r.title.lower() for r in self.experience}

    def redacted(self) -> CandidateProfile:
        """Copy without contact PII, safe to send to an external model."""
        return self.model_copy(update={"email": None, "phone": None})


# --------------------------------------------------------------------- Jobs
class JobQuery(_Frozen):
    roles: list[str] = Field(min_length=1)
    locations: list[str] = Field(default_factory=list)
    work_modes: list[WorkMode] = Field(default_factory=list)
    exp_years: float | None = Field(default=None, ge=0)
    ctc_min_lpa: float | None = Field(default=None, ge=0)
    keywords: list[str] = Field(default_factory=list)
    limit: int = Field(default=50, ge=1, le=500)

    def widened(self) -> JobQuery:
        """Looser query used when no job clears the fit threshold."""
        return self.model_copy(
            update={
                "work_modes": [],
                "ctc_min_lpa": None,
                "keywords": self.keywords[: max(0, len(self.keywords) - 2)],
                "limit": min(self.limit * 2, 500),
            }
        )


class JobPosting(_Frozen):
    source: str
    external_id: str
    title: str
    company: str
    location: str = ""
    locations: list[str] = Field(default_factory=list)  # normalised Indian cities
    url: str = ""
    description: str = ""
    posted_at: datetime | None = None
    work_mode: WorkMode = WorkMode.UNKNOWN
    company_type: CompanyType = CompanyType.UNKNOWN
    ctc_min_lpa: float | None = None
    ctc_max_lpa: float | None = None
    notice_period_days: int | None = None
    exp_min: float | None = None
    exp_max: float | None = None
    required_skills: list[str] = Field(default_factory=list)

    @property
    def job_id(self) -> str:
        return f"{self.source}:{self.external_id}"

    @property
    def content_hash(self) -> str:
        """Cross-source dedupe key: normalised title + company + description head."""
        basis = f"{self.title.strip().lower()}|{self.company.strip().lower()}|"
        basis += " ".join(self.description.lower().split())[:500]
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _ctc_range(self) -> JobPosting:
        if (
            self.ctc_min_lpa is not None
            and self.ctc_max_lpa is not None
            and self.ctc_min_lpa > self.ctc_max_lpa
        ):
            raise ValueError("ctc_min_lpa cannot exceed ctc_max_lpa")
        return self


# ---------------------------------------------------------------------- RAG
class Chunk(_Frozen):
    id: str
    text: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class EvidencePair(_Frozen):
    requirement: str
    evidence: str | None
    evidence_chunk_id: str | None = None
    similarity: float = 0.0


# ------------------------------------------------------------------ Scoring
class FitScore(_Frozen):
    job_id: str
    score: int = Field(ge=0, le=100)
    matched: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    prompt_variant: str = "production"


# ---------------------------------------------------------------- Tailoring
class TailoredResume(_Frozen):
    job_id: str
    summary: str
    bullets: list[str]
    change_log: list[str] = Field(default_factory=list)
    prompt_variant: str = "production"


class ClaimViolation(_Frozen):
    bullet: str
    reason: str


class ReviewDecision(_Frozen):
    job_id: str
    decision: Decision
    edited_summary: str | None = None
    edited_bullets: list[str] | None = None
    comment: str = ""

    @model_validator(mode="after")
    def _edit_needs_content(self) -> ReviewDecision:
        if self.decision is Decision.EDIT and not (self.edited_summary or self.edited_bullets):
            raise ValueError("an 'edit' decision must include edited_summary or edited_bullets")
        return self


# ---------------------------------------------------------------- Interview
class InterviewQuestion(_Frozen):
    question: str
    round_type: RoundType
    skill: str | None = None
    difficulty: int = Field(default=2, ge=1, le=5)


class InterviewPack(_Frozen):
    job_id: str
    questions: list[InterviewQuestion]
    answers: dict[str, str] = Field(default_factory=dict)  # question -> STAR draft

    @field_validator("questions")
    @classmethod
    def _non_empty(cls, v: list[InterviewQuestion]) -> list[InterviewQuestion]:
        if not v:
            raise ValueError("interview pack needs at least one question")
        return v
