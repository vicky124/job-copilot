"""Deterministic eval scorers. LLM-as-judge scorers (pairwise preference) run in
Langfuse experiments and plug in next to these."""

from __future__ import annotations

from copilot.domain.models import CandidateProfile, TailoredResume
from copilot.domain.skills import canonical_skills
from copilot.services.claim_verifier import ClaimVerifier


def faithfulness(profile: CandidateProfile, tailored: TailoredResume) -> float:
    """1.0 when no bullet introduces unsupported facts; minus 1/n per bad line."""
    lines = [tailored.summary, *tailored.bullets]
    bad = {v.bullet for v in ClaimVerifier().verify(profile, tailored)}
    return 1.0 - len(bad) / max(len(lines), 1)


def keyword_coverage(required_skills: list[str], tailored: TailoredResume) -> float:
    """Share of the job's required skills that the tailored text mentions."""
    required = canonical_skills(required_skills)
    if not required:
        return 1.0
    text = " ".join([tailored.summary, *tailored.bullets]).lower()
    return sum(1 for s in required if s in text) / len(required)


def ats_readability(tailored: TailoredResume, max_words: int = 30) -> float:
    """Share of bullets that are ATS-friendly: short, no tables or pipes."""
    if not tailored.bullets:
        return 0.0
    ok = [b for b in tailored.bullets if len(b.split()) <= max_words and "|" not in b]
    return len(ok) / len(tailored.bullets)
