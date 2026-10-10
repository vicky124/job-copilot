"""Structured output: ask the model for JSON, validate with Pydantic, retry once
with the validation error appended."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from copilot.config.settings import Tier
from copilot.ports.llm import LLMClient, LLMMessage

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class StructuredOutputError(ValueError):
    pass


def extract_json(text: str) -> str:
    """Strip code fences and leading prose around the first JSON object."""
    cleaned = _FENCE.sub("", text.strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise StructuredOutputError("no JSON object in model output")
    return cleaned[start : end + 1]


def complete_structured(
    llm: LLMClient,
    tier: Tier,
    system: str,
    user: str,
    schema: type[T],
    *,
    tags: dict[str, str] | None = None,
    retries: int = 1,
) -> T:
    messages = [
        LLMMessage("system", f"{system}\n\nReply with one JSON object only."),
        LLMMessage("user", user),
    ]
    last_error = ""
    for _ in range(retries + 1):
        response = llm.complete(tier, messages, json_mode=True, tags=tags)
        try:
            return schema.model_validate(json.loads(extract_json(response.text)))
        except (StructuredOutputError, json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            messages = [
                *messages,
                LLMMessage("assistant", response.text),
                LLMMessage("user", f"That was invalid: {last_error[:800]}. Return corrected JSON."),
            ]
    raise StructuredOutputError(f"{schema.__name__} parse failed: {last_error}")
