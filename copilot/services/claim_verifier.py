"""No-fabrication guardrail.

A tailored bullet may reorder, rephrase and emphasise. It may not introduce a
skill, employer, number or credential that is not in the original profile.
These checks are deterministic so the guardrail cannot be talked out of.
"""

from __future__ import annotations

import re

from copilot.domain.models import CandidateProfile, ClaimViolation, TailoredResume
from copilot.domain.skills import canonical_skills, extract_skills

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|x|\+|k|cr|crore|lakh|lpa|m|mn|million)?", re.I)
_CREDENTIALS = re.compile(
    r"\b(phd|ph\.d|m\.?tech|b\.?tech|mba|certified|certification|aws certified|pmp|cfa)\b", re.I
)


def _profile_corpus(profile: CandidateProfile) -> str:
    parts = [profile.summary, *profile.skills, *profile.education]
    for role in profile.experience:
        parts += [role.title, role.company, *role.bullets, *role.skills]
    return "\n".join(parts).lower()


def _numbers(text: str) -> set[str]:
    return {m.group(1) for m in _NUMBER.finditer(text)}


class ClaimVerifier:
    def __init__(self, known_companies: set[str] | None = None) -> None:
        """``known_companies``: employer names to watch for (e.g. from the companies config)."""
        self._watch_companies = {c.lower() for c in (known_companies or set())}

    def verify(self, profile: CandidateProfile, tailored: TailoredResume) -> list[ClaimViolation]:
        corpus = _profile_corpus(profile)
        allowed_skills = canonical_skills(profile.all_skills()) | set(extract_skills(corpus))
        allowed_numbers = _numbers(corpus)
        employers = profile.employers()
        violations: list[ClaimViolation] = []

        for text in [tailored.summary, *tailored.bullets]:
            if not text.strip():
                continue
            new_skills = sorted(set(extract_skills(text)) - allowed_skills)
            if new_skills:
                violations.append(ClaimViolation(bullet=text, reason=f"new skills: {new_skills}"))
            new_numbers = sorted(_numbers(text) - allowed_numbers)
            if new_numbers:
                violations.append(
                    ClaimViolation(bullet=text, reason=f"numbers not in profile: {new_numbers}")
                )
            lowered = text.lower()
            for company in self._watch_companies - employers:
                if re.search(rf"\b{re.escape(company)}\b", lowered):
                    violations.append(
                        ClaimViolation(bullet=text, reason=f"employer not in profile: {company}")
                    )
            for cred in {m.group(1).lower() for m in _CREDENTIALS.finditer(text)}:
                if cred not in corpus:
                    violations.append(
                        ClaimViolation(bullet=text, reason=f"credential not in profile: {cred}")
                    )
        return violations
