from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from copilot.domain.models import ApplicationStatus, TailoredResume

# Allowed status transitions. Anything else is a bug or a bad request.
TRANSITIONS: dict[ApplicationStatus, set[ApplicationStatus]] = {
    ApplicationStatus.SAVED: {ApplicationStatus.TAILORED, ApplicationStatus.REJECTED},
    ApplicationStatus.TAILORED: {
        ApplicationStatus.APPROVED,
        ApplicationStatus.TAILORED,
        ApplicationStatus.REJECTED,
    },
    ApplicationStatus.APPROVED: {ApplicationStatus.APPLIED, ApplicationStatus.REJECTED},
    ApplicationStatus.APPLIED: {ApplicationStatus.INTERVIEWING, ApplicationStatus.REJECTED},
    ApplicationStatus.INTERVIEWING: {ApplicationStatus.OFFER, ApplicationStatus.REJECTED},
    ApplicationStatus.OFFER: set(),
    ApplicationStatus.REJECTED: set(),
}


class InvalidTransitionError(ValueError):
    pass


@dataclass(slots=True)
class ApplicationRecord:
    id: str
    user_id: str
    job_id: str
    status: ApplicationStatus = ApplicationStatus.SAVED
    tailored: TailoredResume | None = None
    artifact_uri: str | None = None
    thread_id: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def transition(self, new_status: ApplicationStatus) -> None:
        if new_status not in TRANSITIONS[self.status]:
            raise InvalidTransitionError(f"cannot move {self.status} -> {new_status}")
        self.status = new_status
        self.updated_at = datetime.now(UTC)


@runtime_checkable
class ApplicationRepository(Protocol):
    def get(self, application_id: str) -> ApplicationRecord | None: ...

    def get_by_job(self, user_id: str, job_id: str) -> ApplicationRecord | None: ...

    def save(self, record: ApplicationRecord) -> None: ...

    def list_for_user(
        self, user_id: str, status: ApplicationStatus | None = None
    ) -> list[ApplicationRecord]: ...
