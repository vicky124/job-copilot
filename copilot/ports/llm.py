from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from copilot.config.settings import Tier

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when a model call fails after all fallbacks."""


@runtime_checkable
class LLMClient(Protocol):
    """Chat completion by tier. Callers never name a vendor or model."""

    def complete(
        self,
        tier: Tier,
        messages: list[LLMMessage],
        *,
        json_mode: bool = False,
        tags: dict[str, str] | None = None,
    ) -> LLMResponse: ...
