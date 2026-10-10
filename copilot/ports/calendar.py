from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    title: str
    start: datetime
    end: datetime
    description: str = ""
    attendees: tuple[str, ...] = ()
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("event end must be after start")


@runtime_checkable
class CalendarClient(Protocol):
    def create_event(self, event: CalendarEvent) -> str:
        """Create the event and return its id. Same idempotency_key -> same id."""
        ...
