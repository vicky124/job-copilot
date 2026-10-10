import unittest

from copilot.adapters.llm import (
    BudgetExceededError,
    BudgetGuardLLM,
    FakeLLMClient,
    LiteLLMClient,
    TracedLLM,
    build_llm_client,
)
from copilot.adapters.tracing import InMemoryTracer, NoopTracer, redact_pii
from copilot.ports.llm import LLMError, LLMMessage, LLMResponse
from tests.fakes import offline_settings
from copilot.config.settings import load_settings
from tests.fakes import ROOT

MSG = [LLMMessage("user", "hello")]


def _response(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        "_hidden_params": {"response_cost": 0.001},
    }


class LiteLLMClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_settings(ROOT, env={}, dotenv=False).llm

    def test_calls_tier_model_and_maps_usage(self) -> None:
        seen = {}

        def completion(**kwargs):
            seen.update(kwargs)
            return _response("hi")

        out = LiteLLMClient(self.config, completion).complete("small", MSG, json_mode=True)
        self.assertEqual(seen["model"], "anthropic/claude-haiku-5-5")
        self.assertEqual(seen["response_format"], {"type": "json_object"})
        self.assertEqual((out.text, out.tokens_in, out.tokens_out, out.cost_usd), ("hi", 3, 2, 0.001))

    def test_falls_back_to_next_tier(self) -> None:
        models = []

        def completion(**kwargs):
            models.append(kwargs["model"])
            if "opus" in kwargs["model"]:
                raise TimeoutError("overloaded")
            return _response("ok")

        out = LiteLLMClient(self.config, completion).complete("strong", MSG)
        self.assertEqual(models, ["anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5"])
        self.assertEqual(out.model, "anthropic/claude-sonnet-5-5")

    def test_passes_api_key_from_env_file(self) -> None:
        seen = {}

        def completion(**kwargs):
            seen.update(kwargs)
            return _response("hi")

        LiteLLMClient(self.config, completion, api_keys={"anthropic": "sk-ant-x"}).complete("small", MSG)
        self.assertEqual(seen["api_key"], "sk-ant-x")

    def test_provider_api_keys_skips_dummies(self) -> None:
        from copilot.adapters.llm.factory import provider_api_keys
        from copilot.config.settings import Secrets

        secrets = Secrets.from_env({"ANTHROPIC_API_KEY": "sk-ant-real", "OPENAI_API_KEY": "sk-dummy-0"})
        self.assertEqual(provider_api_keys(secrets), {"anthropic": "sk-ant-real"})

    def test_raises_when_all_fail(self) -> None:
        def completion(**kwargs):
            raise RuntimeError("down")

        with self.assertRaises(LLMError):
            LiteLLMClient(self.config, completion).complete("small", MSG)


class FakeLLMTest(unittest.TestCase):
    def test_rules_and_default(self) -> None:
        llm = FakeLLMClient(default="d").when("hello", "matched")
        self.assertEqual(llm.complete("small", MSG).text, "matched")
        self.assertEqual(llm.complete("small", [LLMMessage("user", "x")]).text, "d")
        self.assertEqual(len(llm.calls), 2)


class DecoratorsTest(unittest.TestCase):
    def test_traced_llm_records_span(self) -> None:
        tracer = InMemoryTracer()
        TracedLLM(FakeLLMClient(default="x"), tracer).complete("medium", MSG, tags={"node": "n1"})
        (span,) = tracer.named("llm.complete")
        self.assertEqual(span.attributes["tier"], "medium")
        self.assertEqual(span.attributes["node"], "n1")
        self.assertEqual(span.attributes["model"], "fake/medium")

    def test_budget_guard_stops_run(self) -> None:
        class Costly:
            def complete(self, tier, messages, *, json_mode=False, tags=None):
                return LLMResponse(text="x", model="m", cost_usd=0.3)

        guard = BudgetGuardLLM(Costly(), max_usd=0.5)
        guard.complete("small", MSG)
        guard.complete("small", MSG)
        self.assertAlmostEqual(guard.spent_usd, 0.6)
        with self.assertRaises(BudgetExceededError):
            guard.complete("small", MSG)

    def test_factory_picks_fake_for_fake_profile(self) -> None:
        llm = build_llm_client(offline_settings().llm, NoopTracer())
        self.assertEqual(llm.complete("small", MSG).model, "fake/small")


class TracingTest(unittest.TestCase):
    def test_in_memory_tracer_records_errors(self) -> None:
        tracer = InMemoryTracer()
        with self.assertRaises(ValueError), tracer.span("boom"):
            raise ValueError("bad")
        self.assertIn("ValueError", tracer.spans[0].error)

    def test_attributes_are_redacted(self) -> None:
        tracer = InMemoryTracer()
        with tracer.span("s", user="asha@example.com") as span:
            span.set_attribute("note", "call +91 98765 43210")
        self.assertEqual(tracer.spans[0].attributes["user"], "[EMAIL]")
        self.assertEqual(tracer.spans[0].attributes["note"], "call [PHONE]")

    def test_redact_pii_nested(self) -> None:
        out = redact_pii({"a": ["PAN ABCDE1234F", "aadhaar 1234 5678 9012"], "n": 5})
        self.assertEqual(out, {"a": ["PAN [PAN]", "aadhaar [AADHAAR]"], "n": 5})


if __name__ == "__main__":
    unittest.main()
