from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from copilot.adapters.tracing.redaction import redact_pii


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None


class NoopTracer:
    """Default when tracing is disabled."""

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[_NoopSpan]:
        yield _NoopSpan()

    def flush(self) -> None:
        return None


@dataclass
class RecordedSpan:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = redact_pii(value)


@dataclass
class InMemoryTracer:
    """Keeps spans in memory. Used in dev mode and by tests to assert on traces."""

    spans: list[RecordedSpan] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[RecordedSpan]:
        recorded = RecordedSpan(name=name, attributes=redact_pii(dict(attributes)))
        started = time.perf_counter()
        try:
            yield recorded
        except Exception as exc:
            recorded.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            recorded.duration_ms = (time.perf_counter() - started) * 1000
            self.spans.append(recorded)

    def flush(self) -> None:
        return None

    def named(self, name: str) -> list[RecordedSpan]:
        return [s for s in self.spans if s.name == name]
