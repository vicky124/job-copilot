"""Langfuse tracer (self-hosted). Imported lazily; enable with TRACING_ENABLED=true."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from copilot.adapters.tracing.redaction import redact_pii


class _LangfuseSpan:
    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def set_attribute(self, key: str, value: Any) -> None:
        self._handle.update(metadata={key: redact_pii(value)})


class LangfuseTracer:
    def __init__(self, host: str, public_key: str, secret_key: str, client: Any = None) -> None:
        if client is None:  # pragma: no cover - needs the langfuse package and a server
            from langfuse import Langfuse  # noqa: PLC0415 - optional dependency

            client = Langfuse(host=host, public_key=public_key, secret_key=secret_key)
        self._client = client

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[_LangfuseSpan]:
        handle = self._client.start_span(name=name, metadata=redact_pii(attributes))
        try:
            yield _LangfuseSpan(handle)
        except Exception as exc:
            handle.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            handle.end()

    def flush(self) -> None:
        self._client.flush()
