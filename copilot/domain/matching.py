"""Pure rules for whether a posting fits a query (used by every job source)."""

from __future__ import annotations

from copilot.domain.india import normalise_cities
from copilot.domain.models import JobPosting, JobQuery, WorkMode

_REMOTE = {WorkMode.REMOTE_INDIA, WorkMode.REMOTE_GLOBAL}


def title_matches(title: str, roles: list[str]) -> bool:
    """Every word of at least one role appears in the title ("backend engineer")."""
    lowered = title.lower()
    return any(all(word in lowered for word in role.lower().split()) for role in roles)


def location_matches(posting: JobPosting, query: JobQuery) -> bool:
    if not query.locations:
        return True
    if posting.work_mode in _REMOTE and (
        not query.work_modes or any(m in _REMOTE for m in query.work_modes)
    ):
        return True
    wanted = {c for loc in query.locations for c in normalise_cities(loc)} or {
        loc.lower() for loc in query.locations
    }
    have = set(posting.locations) or {posting.location.lower()}
    return bool(wanted & have) or any(w.lower() in posting.location.lower() for w in wanted)


def experience_matches(posting: JobPosting, query: JobQuery, slack_years: float = 1.0) -> bool:
    if query.exp_years is None:
        return True
    if posting.exp_min is not None and query.exp_years + slack_years < posting.exp_min:
        return False
    return not (posting.exp_max is not None and query.exp_years - slack_years > posting.exp_max)


def work_mode_matches(posting: JobPosting, query: JobQuery) -> bool:
    if not query.work_modes or posting.work_mode is WorkMode.UNKNOWN:
        return True
    return posting.work_mode in query.work_modes


def ctc_matches(posting: JobPosting, query: JobQuery) -> bool:
    if query.ctc_min_lpa is None or posting.ctc_max_lpa is None:
        return True  # most Indian JDs omit CTC; don't drop them
    return posting.ctc_max_lpa >= query.ctc_min_lpa


def matches_query(posting: JobPosting, query: JobQuery) -> bool:
    return (
        title_matches(posting.title, query.roles)
        and location_matches(posting, query)
        and experience_matches(posting, query)
        and work_mode_matches(posting, query)
        and ctc_matches(posting, query)
    )
