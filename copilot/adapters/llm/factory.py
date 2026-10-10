from __future__ import annotations

from copilot.adapters.llm.decorators import BudgetGuardLLM, TracedLLM
from copilot.adapters.llm.litellm_client import LiteLLMClient
from copilot.adapters.llm.offline import offline_llm
from copilot.config.settings import LLMConfig, Secrets
from copilot.ports.llm import LLMClient
from copilot.ports.tracing import Tracer

# Model prefix in config/llm.yaml -> Secrets field holding its key.
_KEY_FIELDS = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "gemini": "gemini_api_key",
    "azure": "azure_api_key",
}


def provider_api_keys(secrets: Secrets) -> dict[str, str]:
    """Real (non-dummy) keys from .env, keyed by model prefix."""
    keys: dict[str, str] = {}
    for prefix, field in _KEY_FIELDS.items():
        value = getattr(secrets, field)
        if not Secrets.is_dummy(value):
            keys[prefix] = value.get_secret_value()
    return keys


def build_llm_client(config: LLMConfig, tracer: Tracer, secrets: Secrets | None = None) -> LLMClient:
    """Pick the adapter for the configured profile and wrap it.

    Order: Budget(Traced(Base)) so every call is traced, including the one that
    would exceed the budget check on the next turn.
    """
    base: LLMClient = (
        offline_llm()
        if config.provider_profile == "fake"
        else LiteLLMClient(config, api_keys=provider_api_keys(secrets or Secrets()))
    )
    return BudgetGuardLLM(TracedLLM(base, tracer), max_usd=config.budget.per_run_usd)
