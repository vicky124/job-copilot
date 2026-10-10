"""Stdlib HTTP client with retry and backoff. No third-party dependency."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from copilot.ports.http import HttpError

_RETRYABLE = {429, 500, 502, 503, 504}


class UrllibHttpClient:
    def __init__(
        self,
        retries: int = 2,
        backoff_s: float = 0.5,
        user_agent: str = "job-copilot/0.1",
        sleep: Any = time.sleep,
    ) -> None:
        self._retries = retries
        self._backoff = backoff_s
        self._ua = user_agent
        self._sleep = sleep

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 15.0,
    ) -> Any:
        full_url = f"{url}{'&' if '?' in url else '?'}{urllib.parse.urlencode(params)}" if params else url
        request = urllib.request.Request(  # noqa: S310 - URLs come from config, https only
            full_url, headers={"User-Agent": self._ua, "Accept": "application/json", **(headers or {})}
        )
        last_error: HttpError | None = None
        for attempt in range(self._retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = HttpError(url, exc.code, exc.reason)
                if exc.code not in _RETRYABLE:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = HttpError(url, None, str(exc))
            if attempt < self._retries:
                self._sleep(self._backoff * (2**attempt))
        assert last_error is not None
        raise last_error
