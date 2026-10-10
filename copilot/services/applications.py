"""Application tracker use cases (status lifecycle)."""

from __future__ import annotations

import uuid

from copilot.domain.models import ApplicationStatus, TailoredResume
from copilot.ports.repositories import ApplicationRecord, ApplicationRepository


class ApplicationNotFoundError(KeyError):
    pass


class ApplicationService:
    def __init__(self, repo: ApplicationRepository) -> None:
        self._repo = repo

    def save_job(self, user_id: str, job_id: str, thread_id: str | None = None) -> ApplicationRecord:
        existing = self._repo.get_by_job(user_id, job_id)
        if existing:
            return existing
        record = ApplicationRecord(
            id=str(uuid.uuid4()), user_id=user_id, job_id=job_id, thread_id=thread_id
        )
        self._repo.save(record)
        return record

    def record_draft(self, user_id: str, job_id: str, draft: TailoredResume) -> ApplicationRecord:
        record = self.save_job(user_id, job_id)
        record.tailored = draft
        record.transition(ApplicationStatus.TAILORED)
        self._repo.save(record)
        return record

    def approve(self, user_id: str, job_id: str, artifact_uri: str) -> ApplicationRecord:
        record = self._require(user_id, job_id)
        record.artifact_uri = artifact_uri
        record.transition(ApplicationStatus.APPROVED)
        self._repo.save(record)
        return record

    def set_status(
        self, application_id: str, status: ApplicationStatus, user_id: str | None = None
    ) -> ApplicationRecord:
        """Move an application to a new status. With ``user_id``, other users' records
        are reported as not found (no existence leak)."""
        record = self._repo.get(application_id)
        if record is None or (user_id is not None and record.user_id != user_id):
            raise ApplicationNotFoundError(application_id)
        record.transition(status)
        self._repo.save(record)
        return record

    def list(self, user_id: str, status: ApplicationStatus | None = None) -> list[ApplicationRecord]:
        return self._repo.list_for_user(user_id, status)

    def _require(self, user_id: str, job_id: str) -> ApplicationRecord:
        record = self._repo.get_by_job(user_id, job_id)
        if record is None:
            raise ApplicationNotFoundError(f"{user_id}/{job_id}")
        return record
