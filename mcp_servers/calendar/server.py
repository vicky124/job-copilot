"""calendar-mcp: schedule interviews and prep blocks.

v1 writes .ics files (works with Google Calendar, Outlook, Apple). The Google
Calendar adapter (OAuth, GOOGLE_OAUTH_* in .env) implements the same
CalendarClient port and replaces IcsCalendarClient without changing these tools.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from copilot.adapters.calendar import IcsCalendarClient
from copilot.ports.calendar import CalendarClient, CalendarEvent
from mcp_servers.common import create_server, run


def create_event_tool(
    client: CalendarClient,
    title: str,
    start_iso: str,
    end_iso: str,
    notes: str = "",
    attendees: list[str] | None = None,
    idempotency_key: str = "",
) -> dict[str, Any]:
    start, end = datetime.fromisoformat(start_iso), datetime.fromisoformat(end_iso)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end need a timezone offset, e.g. 2026-10-14T10:00:00+05:30")
    event = CalendarEvent(
        title=title,
        start=start,
        end=end,
        description=notes,
        attendees=tuple(attendees or []),
        idempotency_key=idempotency_key,
    )
    return {"event_id": client.create_event(event)}


def build_server() -> Any:
    client = IcsCalendarClient(Path(os.environ.get("APP_OUTPUT_DIR", "outputs")) / "calendar")
    server = create_server("calendar-mcp", "Create interview and prep events. Times in IST (+05:30).")

    @server.tool()
    def create_event(
        title: str,
        start_iso: str,
        end_iso: str,
        notes: str = "",
        attendees: list[str] | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        """Create a calendar event. Repeating the same idempotency_key returns the same id."""
        return create_event_tool(client, title, start_iso, end_iso, notes, attendees, idempotency_key)

    return server


if __name__ == "__main__":  # pragma: no cover
    run(build_server())
