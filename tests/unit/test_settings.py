import unittest

from pydantic import ValidationError

from copilot.config.settings import (
    ConfigError,
    ExperimentConfig,
    LLMConfig,
    Secrets,
    load_settings,
)
from tests.fakes import OFFLINE_ENV, ROOT


class LoadSettingsTest(unittest.TestCase):
    def test_defaults_from_yaml(self) -> None:
        s = load_settings(ROOT, env={}, dotenv=False)
        self.assertEqual(s.llm.provider_profile, "anthropic")
        self.assertEqual(s.llm.embedding_profile, "local")
        self.assertEqual(s.llm.active_embedding.dimensions, 768)
        self.assertEqual(s.vector_store, "qdrant")

    def test_env_overrides_profiles(self) -> None:
        s = load_settings(ROOT, env=OFFLINE_ENV, dotenv=False)
        self.assertEqual(s.llm.provider_profile, "fake")
        self.assertEqual(s.llm.active_embedding.collection_suffix, "v1_hash256")
        self.assertEqual(s.vector_store, "memory")

    def test_unknown_provider_profile_fails_fast(self) -> None:
        with self.assertRaises((ConfigError, ValidationError)):
            load_settings(ROOT, env={"LLM_PROVIDER_PROFILE": "nope"}, dotenv=False)

    def test_model_for_and_fallback_chain(self) -> None:
        llm = load_settings(ROOT, env={}, dotenv=False).llm
        self.assertEqual(llm.model_for("strong").model, "anthropic/claude-opus-5-5")
        chain = [m.model for m in llm.fallback_chain("strong")]
        self.assertEqual(chain, ["anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5"])

    def test_missing_tier_rejected(self) -> None:
        with self.assertRaises((ConfigError, ValidationError)):
            LLMConfig.model_validate(
                {
                    "provider_profile": "p",
                    "profiles": {"p": {"small": {"model": "m"}}},
                    "embedding_profile": "e",
                    "embedding_profiles": {
                        "e": {"model": "x", "dimensions": 8, "collection_suffix": "s"}
                    },
                }
            )

    def test_companies_config_loaded(self) -> None:
        companies = load_settings(ROOT, env={}, dotenv=False).companies
        self.assertEqual(len(companies.active("greenhouse")), 16)
        self.assertEqual({c.slug for c in companies.active("lever")}, {"cred", "meesho", "paytm", "zeta"})
        self.assertEqual(companies.active("workday"), [])


class SecretsTest(unittest.TestCase):
    def test_secrets_masked_in_repr(self) -> None:
        secrets = Secrets.from_env({"ANTHROPIC_API_KEY": "sk-ant-real-value"})
        self.assertNotIn("sk-ant-real-value", repr(secrets))
        self.assertEqual(secrets.anthropic_api_key.get_secret_value(), "sk-ant-real-value")

    def test_dummy_detection(self) -> None:
        self.assertTrue(Secrets.is_dummy(None))
        dummy = Secrets.from_env({"OPENAI_API_KEY": "sk-dummy-000"}).openai_api_key
        self.assertTrue(Secrets.is_dummy(dummy))

    def test_prod_refuses_dummy_secrets(self) -> None:
        env = {"APP_ENV": "prod", "APP_JWT_SECRET": "dummy-jwt", "ANTHROPIC_API_KEY": "dummy"}
        with self.assertRaises(ConfigError):
            load_settings(ROOT, env=env, dotenv=False)

    def test_prod_accepts_real_secrets(self) -> None:
        env = {"APP_ENV": "prod", "APP_JWT_SECRET": "s3cret", "ANTHROPIC_API_KEY": "sk-ant-abc"}
        self.assertEqual(load_settings(ROOT, env=env, dotenv=False).env, "prod")


class ExperimentConfigTest(unittest.TestCase):
    def test_weights_must_sum_to_one(self) -> None:
        with self.assertRaises((ConfigError, ValidationError)):
            ExperimentConfig(variants={"a": 0.5, "b": 0.6})


if __name__ == "__main__":
    unittest.main()
