from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Span(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    """Tracing abstraction. Langfuse in production, no-op or recording in tests."""

    def span(self, name: str, **attributes: Any) -> AbstractContextManager[Span]: ...

    def flush(self) -> None: ...
