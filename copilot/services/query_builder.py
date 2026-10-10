"""Profile + user preferences -> JobQuery. Deterministic; no model call needed."""

from __future__ import annotations

from pydantic import BaseModel, Field

from copilot.domain.models import CandidateProfile, JobQuery, WorkMode


class SearchPreferences(BaseModel):
    roles: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    work_modes: list[WorkMode] = Field(default_factory=list)
    min_ctc_lpa: float | None = None
    limit: int = 50


class QueryBuilder:
    def build(self, profile: CandidateProfile, prefs: SearchPreferences) -> JobQuery:
        roles = prefs.roles or self._roles_from_profile(profile)
        if not roles:
            raise ValueError("no target roles: set preferences.roles or add experience")
        keywords = sorted(profile.all_skills())[:8]
        return JobQuery(
            roles=roles,
            locations=prefs.locations,
            work_modes=prefs.work_modes,
            exp_years=profile.total_years or None,
            ctc_min_lpa=prefs.min_ctc_lpa
            if prefs.min_ctc_lpa is not None
            else profile.expected_ctc_lpa,
            keywords=keywords,
            limit=prefs.limit,
        )

    @staticmethod
    def _roles_from_profile(profile: CandidateProfile) -> list[str]:
        if not profile.experience:
            return []
        latest = profile.experience[0].title
        # Drop seniority words so "Senior Backend Engineer" also matches "Backend Engineer II".
        stop = {"senior", "sr", "sr.", "junior", "jr", "lead", "principal", "staff", "ii", "iii"}
        core = " ".join(w for w in latest.split() if w.lower() not in stop)
        return [core or latest]
