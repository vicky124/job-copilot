"""Compile GRAPH_SPEC into a LangGraph StateGraph.

Requires ``pip install 'job-copilot[graph]'``. In production pass a
``PostgresSaver`` checkpointer so a run paused at human_review survives restarts;
resume with ``graph.invoke(Command(resume=decisions), config)``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from copilot.config.settings import PipelineConfig
from copilot.graph.nodes import PipelineNodes
from copilot.graph.spec import END, GRAPH_SPEC, START_NODE, GraphSpec
from copilot.graph.state import CopilotState


def build_graph(
    nodes: PipelineNodes,
    config: PipelineConfig,
    checkpointer: Any = None,
    spec: GraphSpec = GRAPH_SPEC,
) -> Any:
    try:
        from langgraph.graph import END as LG_END  # noqa: PLC0415 - optional dependency
        from langgraph.graph import START, StateGraph  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError("langgraph not installed: pip install 'job-copilot[graph]'") from exc

    def _target(name: str) -> str:
        return LG_END if name == END else name

    graph = StateGraph(CopilotState)
    for name, fn in nodes.as_mapping().items():
        graph.add_node(name, fn)
    graph.add_edge(START, START_NODE)
    for source, target in spec.edges.items():
        graph.add_edge(source, _target(target))
    for source, (router, allowed) in spec.conditional.items():
        graph.add_conditional_edges(
            source,
            partial(_route, router=router, config=config),
            {name: _target(name) for name in allowed},
        )
    return graph.compile(checkpointer=checkpointer)


def _route(state: CopilotState, *, router: Any, config: PipelineConfig) -> str:
    return str(router(state, config))
