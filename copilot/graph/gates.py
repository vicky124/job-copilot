"""Human-in-the-loop as a port.

``human_review`` asks a ReviewGate for decisions. In LangGraph the gate calls
``interrupt()`` (the run pauses and resumes with ``Command(resume=...)``); locally
it can raise ReviewPending, call a callback, or auto-approve in tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from copilot.domain.models import Decision, ReviewDecision


class ReviewPending(Exception):  # noqa: N818 - control-flow signal, not an error
    """Raised to pause a local run until decisions arrive."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("human review required")
        self.payload = payload


@runtime_checkable
class ReviewGate(Protocol):
    def request(self, payload: dict[str, Any]) -> dict[str, ReviewDecision]: ...


def parse_decisions(raw: Any) -> dict[str, ReviewDecision]:
    """Accept {job_id: ReviewDecision | dict} or a list of dicts."""
    items = raw.values() if isinstance(raw, dict) else raw or []
    decisions = [d if isinstance(d, ReviewDecision) else ReviewDecision.model_validate(d) for d in items]
    return {d.job_id: d for d in decisions}


class LangGraphInterruptGate:
    def request(self, payload: dict[str, Any]) -> dict[str, ReviewDecision]:  # pragma: no cover
        from langgraph.types import interrupt  # noqa: PLC0415 - optional dependency

        return parse_decisions(interrupt(payload))


class CallbackReviewGate:
    def __init__(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        self._callback = callback

    def request(self, payload: dict[str, Any]) -> dict[str, ReviewDecision]:
        return parse_decisions(self._callback(payload))


class PausingReviewGate:
    """Local equivalent of interrupt(): pause unless decisions were supplied."""

    def __init__(self, decisions: dict[str, ReviewDecision] | None = None) -> None:
        self.decisions = decisions or {}

    def request(self, payload: dict[str, Any]) -> dict[str, ReviewDecision]:
        """Each decision answers one review round, like a resumed interrupt().
        Decisions are consumed, so an 'edit' leads to a fresh pause for re-review."""
        pending = [job["job_id"] for job in payload["drafts"]]
        if not all(job_id in self.decisions for job_id in pending):
            raise ReviewPending(payload)
        return {job_id: self.decisions.pop(job_id) for job_id in pending}


class AutoApproveGate:
    """Tests and demos only. Never wire this in production."""

    def request(self, payload: dict[str, Any]) -> dict[str, ReviewDecision]:
        return {
            job["job_id"]: ReviewDecision(job_id=job["job_id"], decision=Decision.APPROVE)
            for job in payload["drafts"]
        }
