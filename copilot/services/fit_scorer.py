"""Fit score (0-100).

The number is computed deterministically from evidence, so it is explainable and
stable across prompt changes. The LLM (medium tier) only writes the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

from copilot.domain.models import CandidateProfile, EvidencePair, FitScore, JobPosting
from copilot.domain.skills import canonical_skills, extract_skills
from copilot.ports.llm import LLMClient, LLMMessage
from copilot.ports.prompts import PromptStore


@dataclass(frozen=True)
class ScoringWeights:
    skills: float = 50.0
    evidence: float = 25.0
    experience: float = 15.0
    notice_period: float = 5.0
    ctc: float = 5.0

    @property
    def total(self) -> float:
        return self.skills + self.evidence + self.experience + self.notice_period + self.ctc


def experience_fit(years: float, exp_min: float | None, exp_max: float | None) -> float:
    """1.0 inside the band, decaying 0.25 per year outside it."""
    if exp_min is None and exp_max is None:
        return 1.0
    if exp_min is not None and years < exp_min:
        return max(0.0, 1.0 - 0.25 * (exp_min - years))
    if exp_max is not None and years > exp_max:
        return max(0.0, 1.0 - 0.25 * (years - exp_max))
    return 1.0


def notice_fit(candidate_days: int | None, max_days: int | None) -> float:
    if candidate_days is None or max_days is None or candidate_days <= max_days:
        return 1.0
    return max(0.0, 1.0 - (candidate_days - max_days) / 60)


def ctc_fit(expected: float | None, ctc_max: float | None) -> float:
    if expected is None or ctc_max is None or expected <= ctc_max:
        return 1.0
    return max(0.0, 1.0 - (expected - ctc_max) / max(ctc_max, 1.0))


class FitScorer:
    PROMPT = "score_fit"

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        llm: LLMClient | None = None,
        prompts: PromptStore | None = None,
    ) -> None:
        self._w = weights or ScoringWeights()
        self._llm = llm
        self._prompts = prompts

    def score(
        self,
        profile: CandidateProfile,
        posting: JobPosting,
        evidence: list[EvidencePair],
        prompt_variant: str = "production",
    ) -> FitScore:
        required = canonical_skills(posting.required_skills) or set(
            extract_skills(" ".join(e.requirement for e in evidence))
        )
        have = canonical_skills(profile.all_skills())
        matched = sorted(required & have)
        gaps = sorted(required - have)

        skill_ratio = len(matched) / len(required) if required else 0.5
        evidence_ratio = (
            sum(1 for e in evidence if e.evidence) / len(evidence) if evidence else 0.0
        )
        w = self._w
        raw = (
            w.skills * skill_ratio
            + w.evidence * evidence_ratio
            + w.experience * experience_fit(profile.total_years, posting.exp_min, posting.exp_max)
            + w.notice_period
            * notice_fit(profile.notice_period_days, posting.notice_period_days)
            + w.ctc * ctc_fit(profile.expected_ctc_lpa, posting.ctc_max_lpa)
        )
        score = round(100 * raw / w.total)
        rationale = self._rationale(posting, matched, gaps, evidence_ratio, prompt_variant)
        return FitScore(
            job_id=posting.job_id,
            score=max(0, min(100, score)),
            matched=matched,
            gaps=gaps,
            evidence_ids=[e.evidence_chunk_id for e in evidence if e.evidence_chunk_id],
            rationale=rationale,
            prompt_variant=prompt_variant,
        )

    def _rationale(
        self,
        posting: JobPosting,
        matched: list[str],
        gaps: list[str],
        evidence_ratio: float,
        variant: str,
    ) -> str:
        default = (
            f"Matches {len(matched)} of {len(matched) + len(gaps)} required skills"
            f"{' (missing: ' + ', '.join(gaps[:5]) + ')' if gaps else ''}; "
            f"evidence found for {evidence_ratio:.0%} of requirements."
        )
        if self._llm is None or self._prompts is None:
            return default
        prompt = self._prompts.get(self.PROMPT, variant)
        system, user = prompt.render(
            title=posting.title,
            company=posting.company,
            matched=", ".join(matched) or "none",
            gaps=", ".join(gaps) or "none",
            evidence_ratio=f"{evidence_ratio:.0%}",
        )
        response = self._llm.complete(
            "medium",
            [LLMMessage("system", system), LLMMessage("user", user)],
            tags={"node": "score_fit", "prompt_version": prompt.version, "prompt_variant": variant},
        )
        return response.text.strip() or default
