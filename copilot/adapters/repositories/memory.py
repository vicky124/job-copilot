"""In-memory repository. The Postgres implementation (SQLAlchemy) implements the
same ApplicationRepository port; nothing above this layer changes when swapping."""

from __future__ import annotations

import copy
import threading

from copilot.domain.models import ApplicationStatus
from copilot.ports.repositories import ApplicationRecord


class InMemoryApplicationRepository:
    def __init__(self) -> None:
        self._items: dict[str, ApplicationRecord] = {}
        self._lock = threading.Lock()

    def get(self, application_id: str) -> ApplicationRecord | None:
        with self._lock:
            item = self._items.get(application_id)
            return copy.deepcopy(item) if item else None

    def get_by_job(self, user_id: str, job_id: str) -> ApplicationRecord | None:
        with self._lock:
            for item in self._items.values():
                if item.user_id == user_id and item.job_id == job_id:
                    return copy.deepcopy(item)
        return None

    def save(self, record: ApplicationRecord) -> None:
        with self._lock:
            self._items[record.id] = copy.deepcopy(record)

    def list_for_user(
        self, user_id: str, status: ApplicationStatus | None = None
    ) -> list[ApplicationRecord]:
        with self._lock:
            items = [
                copy.deepcopy(i)
                for i in self._items.values()
                if i.user_id == user_id and (status is None or i.status == status)
            ]
        return sorted(items, key=lambda r: r.updated_at, reverse=True)
