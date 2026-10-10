"""Decorators that add cross-cutting behaviour to any LLMClient.

They wrap the port instead of changing adapters (Open/Closed): tracing and
budgets work the same whether the inner client is LiteLLM or the fake.
"""

from __future__ import annotations

import threading

from copilot.config.settings import Tier
from copilot.ports.llm import LLMClient, LLMMessage, LLMResponse
from copilot.ports.tracing import Tracer


class TracedLLM:
    """Records one span per model call with tier, model, tokens, cost and tags."""

    def __init__(self, inner: LLMClient, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer

    def complete(
        self,
        tier: Tier,
        messages: list[LLMMessage],
        *,
        json_mode: bool = False,
        tags: dict[str, str] | None = None,
    ) -> LLMResponse:
        tags = tags or {}
        with self._tracer.span("llm.complete", tier=tier, **tags) as span:
            response = self._inner.complete(tier, messages, json_mode=json_mode, tags=tags)
            span.set_attribute("model", response.model)
            span.set_attribute("tokens_in", response.tokens_in)
            span.set_attribute("tokens_out", response.tokens_out)
            span.set_attribute("cost_usd", response.cost_usd)
            return response


class BudgetExceededError(RuntimeError):
    pass


class BudgetGuardLLM:
    """Stops a run once its LLM spend passes the configured ceiling."""

    def __init__(self, inner: LLMClient, max_usd: float) -> None:
        self._inner = inner
        self._max_usd = max_usd
        self._spent = 0.0
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        return self._spent

    def complete(
        self,
        tier: Tier,
        messages: list[LLMMessage],
        *,
        json_mode: bool = False,
        tags: dict[str, str] | None = None,
    ) -> LLMResponse:
        with self._lock:
            if self._spent >= self._max_usd:
                raise BudgetExceededError(
                    f"run budget of ${self._max_usd:.2f} exhausted (spent ${self._spent:.4f})"
                )
        response = self._inner.complete(tier, messages, json_mode=json_mode, tags=tags)
        with self._lock:
            self._spent += response.cost_usd
        return response
