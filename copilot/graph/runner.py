"""Framework-free executor for GRAPH_SPEC.

Runs the exact same nodes and routers as the LangGraph build. Used for local
dev without infrastructure and for end-to-end tests. Production uses LangGraph
(checkpointing, interrupts that survive restarts, parallel fan-out).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.config.settings import PipelineConfig
from copilot.graph.gates import ReviewPending
from copilot.graph.nodes import PipelineNodes
from copilot.graph.spec import END, GRAPH_SPEC, START_NODE, GraphSpec
from copilot.graph.state import CopilotState, apply_update
from copilot.ports.tracing import Tracer


class RecursionLimitError(RuntimeError):
    pass


@dataclass
class RunResult:
    state: CopilotState
    path: list[str] = field(default_factory=list)
    paused_at: str | None = None
    review_payload: dict[str, object] | None = None

    @property
    def finished(self) -> bool:
        return self.paused_at is None


class LocalPipelineRunner:
    def __init__(
        self,
        nodes: PipelineNodes,
        config: PipelineConfig,
        tracer: Tracer,
        spec: GraphSpec = GRAPH_SPEC,
        recursion_limit: int = 50,
    ) -> None:
        self._nodes = nodes.as_mapping()
        self._cfg = config
        self._tracer = tracer
        self._spec = spec
        self._limit = recursion_limit

    def _next(self, node: str, state: CopilotState) -> str:
        if node in self._spec.conditional:
            router, allowed = self._spec.conditional[node]
            target = router(state, self._cfg)
            if target not in allowed:
                raise RuntimeError(f"router for {node} returned {target}, allowed {allowed}")
            return target
        return self._spec.edges[node]

    def run(self, state: CopilotState, start: str = START_NODE) -> RunResult:
        """Run from ``start`` until END or a review pause.

        Resume after a pause with ``run(result.state, start="human_review")``
        once the gate has decisions.
        """
        node, path = start, []
        with self._tracer.span("pipeline.run", user_id=state.get("user_id", "")):
            while node != END:
                if len(path) >= self._limit:
                    raise RecursionLimitError(f"exceeded {self._limit} steps: {path[-6:]}")
                path.append(node)
                try:
                    with self._tracer.span(f"node.{node}"):
                        update = self._nodes[node](state)
                except ReviewPending as pause:
                    return RunResult(
                        state=state, path=path, paused_at=node, review_payload=pause.payload
                    )
                state = apply_update(state, update)
                node = self._next(node, state)
        return RunResult(state=state, path=path)
