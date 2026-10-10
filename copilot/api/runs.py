"""Run engines: start a pipeline run, pause at human review, resume with decisions.

``LocalRunEngine`` keeps paused state in memory (dev, tests).
``LangGraphRunEngine`` uses LangGraph + a checkpointer (production): the paused
run lives in Postgres and survives restarts.
The API depends only on the RunEngine protocol.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from copilot.domain.models import ReviewDecision
from copilot.graph.gates import PausingReviewGate
from copilot.graph.nodes import PipelineNodes, PipelineServices
from copilot.graph.runner import LocalPipelineRunner
from copilot.graph.state import CopilotState

RunStatus = Literal["awaiting_review", "completed", "failed"]


class RunNotFoundError(KeyError):
    pass


class RunStateError(RuntimeError):
    pass


@dataclass
class RunSnapshot:
    thread_id: str
    status: RunStatus
    state: CopilotState
    review_payload: dict[str, Any] | None = None
    path: list[str] = field(default_factory=list)
    error: str | None = None


class RunEngine(Protocol):
    def start(self, state: CopilotState) -> RunSnapshot: ...

    def resume(self, thread_id: str, decisions: list[ReviewDecision]) -> RunSnapshot: ...

    def get(self, thread_id: str) -> RunSnapshot: ...


class LocalRunEngine:
    def __init__(self, services: PipelineServices) -> None:
        self._services = services
        self._runs: dict[str, RunSnapshot] = {}
        self._lock = threading.Lock()

    def _execute(self, thread_id: str, state: CopilotState, gate: PausingReviewGate, start: str) -> RunSnapshot:
        runner = LocalPipelineRunner(
            PipelineNodes(self._services, gate), self._services.pipeline, self._services.tracer
        )
        try:
            result = runner.run(state, start=start)
            snapshot = RunSnapshot(
                thread_id=thread_id,
                status="completed" if result.finished else "awaiting_review",
                state=result.state,
                review_payload=result.review_payload,
                path=result.path,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a failed run, keep the server up
            snapshot = RunSnapshot(thread_id, "failed", state, error=f"{type(exc).__name__}: {exc}")
        with self._lock:
            self._runs[thread_id] = snapshot
        return snapshot

    def start(self, state: CopilotState) -> RunSnapshot:
        return self._execute(str(uuid.uuid4()), state, PausingReviewGate(), "parse_resume")

    def resume(self, thread_id: str, decisions: list[ReviewDecision]) -> RunSnapshot:
        snapshot = self.get(thread_id)
        if snapshot.status != "awaiting_review" or snapshot.review_payload is None:
            raise RunStateError(f"run {thread_id} is {snapshot.status}, not awaiting review")
        pending = {d["job_id"] for d in snapshot.review_payload["drafts"]}
        given = {d.job_id: d for d in decisions}
        missing = pending - given.keys()
        if missing:
            raise RunStateError(f"missing decisions for jobs: {sorted(missing)}")
        gate = PausingReviewGate({k: v for k, v in given.items() if k in pending})
        return self._execute(thread_id, snapshot.state, gate, "human_review")

    def get(self, thread_id: str) -> RunSnapshot:
        with self._lock:
            if thread_id not in self._runs:
                raise RunNotFoundError(thread_id)
            return self._runs[thread_id]


class LangGraphRunEngine:  # pragma: no cover - needs langgraph and a checkpointer
    """Production engine. ``graph`` is from copilot.graph.builder.build_graph(...)
    compiled with a PostgresSaver checkpointer and nodes using LangGraphInterruptGate."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def _snapshot(self, thread_id: str) -> RunSnapshot:
        config = {"configurable": {"thread_id": thread_id}}
        snap = self._graph.get_state(config)
        interrupts = [i for task in snap.tasks for i in task.interrupts]
        return RunSnapshot(
            thread_id=thread_id,
            status="awaiting_review" if interrupts else "completed",
            state=snap.values,
            review_payload=interrupts[0].value if interrupts else None,
        )

    def start(self, state: CopilotState) -> RunSnapshot:
        thread_id = str(uuid.uuid4())
        self._graph.invoke(state, {"configurable": {"thread_id": thread_id}})
        return self._snapshot(thread_id)

    def resume(self, thread_id: str, decisions: list[ReviewDecision]) -> RunSnapshot:
        from langgraph.types import Command  # noqa: PLC0415

        payload = [d.model_dump(mode="json") for d in decisions]
        self._graph.invoke(Command(resume=payload), {"configurable": {"thread_id": thread_id}})
        return self._snapshot(thread_id)

    def get(self, thread_id: str) -> RunSnapshot:
        return self._snapshot(thread_id)
