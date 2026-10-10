from __future__ import annotations

from typing import Protocol, runtime_checkable

from copilot.domain.models import JobPosting, JobQuery


@runtime_checkable
class JobSource(Protocol):
    """One place jobs come from (Adzuna, a Greenhouse board, pasted text...).

    New sources are added by implementing this protocol — no change to the
    aggregator or graph (Open/Closed).
    """

    @property
    def name(self) -> str: ...

    def search(self, query: JobQuery) -> list[JobPosting]: ...
