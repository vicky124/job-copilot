"""ICS file calendar: the fallback until Google/Outlook OAuth is wired in calendar-mcp.
Writes one .ics per event that the user can import into any calendar app."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from copilot.ports.calendar import CalendarEvent


def _fmt(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")
    )


def event_uid(event: CalendarEvent) -> str:
    basis = event.idempotency_key or f"{event.title}|{event.start.isoformat()}"
    return f"{hashlib.sha256(basis.encode()).hexdigest()[:24]}@job-copilot"


def to_ics(event: CalendarEvent, now: datetime | None = None) -> str:
    stamp = _fmt(now or datetime.now(UTC))
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//job-copilot//EN",
        "BEGIN:VEVENT",
        f"UID:{event_uid(event)}",
        f"DTSTAMP:{stamp}",
        f"DTSTART:{_fmt(event.start)}",
        f"DTEND:{_fmt(event.end)}",
        f"SUMMARY:{_escape(event.title)}",
    ]
    if event.description:
        lines.append(f"DESCRIPTION:{_escape(event.description)}")
    lines.extend(f"ATTENDEE:mailto:{a}" for a in event.attendees)
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(lines) + "\r\n"


class IcsCalendarClient:
    def __init__(self, output_dir: Path) -> None:
        self._dir = output_dir

    def create_event(self, event: CalendarEvent) -> str:
        uid = event_uid(event)
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{uid.split('@')[0]}.ics"
        if not path.exists():  # idempotent: same key -> same file, written once
            path.write_text(to_ics(event), encoding="utf-8")
        return uid
