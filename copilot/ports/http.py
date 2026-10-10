from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class HttpError(RuntimeError):
    def __init__(self, url: str, status: int | None, message: str = "") -> None:
        super().__init__(f"HTTP {status} for {url}: {message}".strip())
        self.url = url
        self.status = status


@runtime_checkable
class HttpClient(Protocol):
    """Minimal JSON GET client so job sources stay testable without the network."""

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 15.0,
    ) -> Any: ...
