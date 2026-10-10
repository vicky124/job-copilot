"""LiteLLM adapter: one gateway for Anthropic, OpenAI, Gemini, Azure and Ollama.

Provider and model per tier come from ``config/llm.yaml``; credentials come from
the standard env vars LiteLLM reads (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...).
"""

from __future__ import annotations

import logging
from typing import Any

from copilot.config.settings import LLMConfig, ModelSpec, Tier
from copilot.ports.llm import LLMError, LLMMessage, LLMResponse

log = logging.getLogger(__name__)


class LiteLLMClient:
    def __init__(
        self,
        config: LLMConfig,
        completion_fn: Any | None = None,
        api_keys: dict[str, str] | None = None,
    ) -> None:
        """``completion_fn`` defaults to ``litellm.completion``; injectable for tests.

        ``api_keys`` maps a model prefix ("anthropic", "openai", ...) to its key, so
        keys from .env work without exporting them as process environment variables.
        """
        self._config = config
        self._completion = completion_fn
        self._api_keys = api_keys or {}

    def _fn(self) -> Any:
        if self._completion is None:
            try:
                import litellm  # noqa: PLC0415 - optional dependency
            except ImportError as exc:  # pragma: no cover - depends on env
                raise LLMError("litellm not installed: pip install 'job-copilot[llm]'") from exc
            self._completion = litellm.completion
        return self._completion

    def complete(
        self,
        tier: Tier,
        messages: list[LLMMessage],
        *,
        json_mode: bool = False,
        tags: dict[str, str] | None = None,
    ) -> LLMResponse:
        errors: list[str] = []
        for spec in self._config.fallback_chain(tier):
            try:
                return self._call(spec, messages, json_mode=json_mode, tags=tags or {})
            except Exception as exc:  # noqa: BLE001 - try the next model in the chain
                log.warning("model %s failed (%s); trying fallback", spec.model, exc)
                errors.append(f"{spec.model}: {exc}")
        raise LLMError(f"all models failed for tier '{tier}': {'; '.join(errors)}")

    def _call(
        self,
        spec: ModelSpec,
        messages: list[LLMMessage],
        *,
        json_mode: bool,
        tags: dict[str, str],
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": spec.temperature,
            "metadata": tags,
        }
        if spec.api_base:
            kwargs["api_base"] = spec.api_base
        if key := self._api_keys.get(spec.model.split("/", 1)[0]):
            kwargs["api_key"] = key
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        raw = self._fn()(**kwargs)
        usage = _get(raw, "usage") or {}
        hidden = _get(raw, "_hidden_params") or {}
        return LLMResponse(
            text=_get(_get(_get(raw, "choices")[0], "message"), "content") or "",
            model=spec.model,
            tokens_in=int(_get(usage, "prompt_tokens") or 0),
            tokens_out=int(_get(usage, "completion_tokens") or 0),
            cost_usd=float(_get(hidden, "response_cost") or 0.0),
        )


def _get(obj: Any, key: str) -> Any:
    """Read a field from a LiteLLM response object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
