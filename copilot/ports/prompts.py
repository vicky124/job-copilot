from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class PromptNotFoundError(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    version: str
    label: str
    system: str
    user: str

    def render(self, **variables: str) -> tuple[str, str]:
        """Fill ``{placeholders}``. Missing variables raise KeyError (fail loudly)."""
        return self.system.format(**variables), self.user.format(**variables)


@runtime_checkable
class PromptStore(Protocol):
    """Files in prompts/ for dev; Langfuse prompt management in staging and prod."""

    def get(self, name: str, label: str = "production") -> Prompt: ...
