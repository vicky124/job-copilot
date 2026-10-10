"""Tailor resume summary and bullets to a job (strong tier)."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from copilot.domain.models import (
    CandidateProfile,
    ClaimViolation,
    EvidencePair,
    JobPosting,
    ReviewDecision,
    TailoredResume,
)
from copilot.ports.llm import LLMClient
from copilot.ports.prompts import PromptStore
from copilot.services.structured import complete_structured


class _TailorOutput(BaseModel):
    summary: str
    bullets: list[str] = Field(min_length=1, max_length=12)
    change_log: list[str] = Field(default_factory=list)


class TailorService:
    PROMPT = "tailor_resume"

    def __init__(self, llm: LLMClient, prompts: PromptStore) -> None:
        self._llm = llm
        self._prompts = prompts

    def tailor(
        self,
        profile: CandidateProfile,
        posting: JobPosting,
        evidence: list[EvidencePair],
        *,
        prompt_variant: str = "production",
        violations: list[ClaimViolation] | None = None,
    ) -> TailoredResume:
        prompt = self._prompts.get(self.PROMPT, prompt_variant)
        safe = profile.redacted()
        original_bullets = [b for role in safe.experience for b in role.bullets]
        feedback = (
            "Fix these problems from your previous draft:\n"
            + "\n".join(f"- {v.reason}: {v.bullet}" for v in violations)
            if violations
            else "None."
        )
        system, user = prompt.render(
            job_title=posting.title,
            company=posting.company,
            requirements="\n".join(f"- {e.requirement}" for e in evidence) or posting.description[:3000],
            evidence=json.dumps(
                [{"requirement": e.requirement, "evidence": e.evidence} for e in evidence],
                ensure_ascii=False,
            ),
            summary=safe.summary,
            bullets="\n".join(f"- {b}" for b in original_bullets),
            skills=", ".join(safe.skills),
            feedback=feedback,
        )
        out = complete_structured(
            self._llm,
            "strong",
            system,
            user,
            _TailorOutput,
            tags={
                "node": "tailor_resume",
                "job_id": posting.job_id,
                "prompt_version": prompt.version,
                "prompt_variant": prompt_variant,
            },
        )
        return TailoredResume(
            job_id=posting.job_id,
            summary=out.summary,
            bullets=out.bullets,
            change_log=out.change_log,
            prompt_variant=prompt_variant,
        )

    @staticmethod
    def apply_edits(draft: TailoredResume, decision: ReviewDecision) -> TailoredResume:
        """Merge the user's edits from the review screen into the draft."""
        return draft.model_copy(
            update={
                "summary": decision.edited_summary or draft.summary,
                "bullets": decision.edited_bullets or draft.bullets,
                "change_log": [*draft.change_log, "edited by user"],
            }
        )
