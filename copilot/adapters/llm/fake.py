"""Deterministic offline LLM for local dev (provider_profile: fake) and tests.

Responses are chosen by the first rule whose trigger appears in the prompt.
With no rule matched it returns ``default``. Every call is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.config.settings import Tier
from copilot.ports.llm import LLMMessage, LLMResponse


@dataclass
class FakeCall:
    tier: Tier
    messages: list[LLMMessage]
    json_mode: bool
    tags: dict[str, str]


@dataclass
class FakeLLMClient:
    rules: list[tuple[str, str]] = field(default_factory=list)
    default: str = "{}"
    calls: list[FakeCall] = field(default_factory=list)

    def when(self, trigger: str, respond: str) -> FakeLLMClient:
        """Add a rule: if ``trigger`` is in the prompt, reply with ``respond``."""
        self.rules.append((trigger, respond))
        return self

    def complete(
        self,
        tier: Tier,
        messages: list[LLMMessage],
        *,
        json_mode: bool = False,
        tags: dict[str, str] | None = None,
    ) -> LLMResponse:
        self.calls.append(FakeCall(tier, list(messages), json_mode, dict(tags or {})))
        prompt = "\n".join(m.content for m in messages)
        text = next((reply for trigger, reply in self.rules if trigger in prompt), self.default)
        return LLMResponse(
            text=text,
            model=f"fake/{tier}",
            tokens_in=len(prompt.split()),
            tokens_out=len(text.split()),
        )
