"""Typed configuration.

Two sources, kept apart on purpose:
  * YAML files in ``config/`` hold non-secret tuning (models per tier, thresholds,
    company boards). They are committed.
  * Environment variables (``.env``) hold secrets and per-environment overrides.
    They are never committed; ``.env.example`` documents every key with a dummy value.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator

Tier = Literal["small", "medium", "strong"]
TIERS: tuple[Tier, ...] = ("small", "medium", "strong")


class ConfigError(ValueError):
    """Raised when configuration is missing or inconsistent."""


# --------------------------------------------------------------------------- LLM
class ModelSpec(BaseModel):
    model: str
    temperature: float = 0.0
    api_base: str | None = None


class EmbeddingProfile(BaseModel):
    model: str
    dimensions: int = Field(gt=0)
    collection_suffix: str
    api_base: str | None = None
    doc_prefix: str = ""
    query_prefix: str = ""


class Budget(BaseModel):
    per_run_usd: float = 0.5
    per_user_daily_usd: float = 2.0


class LLMConfig(BaseModel):
    provider_profile: str
    profiles: dict[str, dict[Tier, ModelSpec]]
    fallbacks: dict[Tier, list[Tier]] = Field(default_factory=dict)
    embedding_profile: str
    embedding_profiles: dict[str, EmbeddingProfile]
    budget: Budget = Field(default_factory=Budget)

    @model_validator(mode="after")
    def _check_profiles(self) -> LLMConfig:
        if self.provider_profile not in self.profiles:
            raise ConfigError(
                f"provider_profile '{self.provider_profile}' not in profiles "
                f"{sorted(self.profiles)}"
            )
        missing = [t for t in TIERS if t not in self.profiles[self.provider_profile]]
        if missing:
            raise ConfigError(f"profile '{self.provider_profile}' missing tiers {missing}")
        if self.embedding_profile not in self.embedding_profiles:
            raise ConfigError(
                f"embedding_profile '{self.embedding_profile}' not in embedding_profiles "
                f"{sorted(self.embedding_profiles)}"
            )
        return self

    def model_for(self, tier: Tier) -> ModelSpec:
        return self.profiles[self.provider_profile][tier]

    def fallback_chain(self, tier: Tier) -> list[ModelSpec]:
        """The requested tier first, then its configured fallbacks."""
        chain = [self.model_for(tier)]
        chain.extend(self.model_for(t) for t in self.fallbacks.get(tier, []))
        return chain

    @property
    def active_embedding(self) -> EmbeddingProfile:
        return self.embedding_profiles[self.embedding_profile]


# --------------------------------------------------------------------------- App
class PipelineConfig(BaseModel):
    fit_threshold: int = Field(default=60, ge=0, le=100)
    max_jobs_to_tailor: int = Field(default=5, ge=1)
    max_query_widen_loops: int = Field(default=2, ge=0)
    max_tailor_retries: int = Field(default=2, ge=0)
    require_human_approval: bool = True


class RetrievalConfig(BaseModel):
    top_k: int = 30
    keep_after_rerank: int = 8
    min_score: float = 0.25
    chunk_tokens: int = 400
    chunk_overlap: int = 50


class CollectionsConfig(BaseModel):
    job_descriptions: str = "job_descriptions"
    interview_questions: str = "interview_questions"
    candidate_profile: str = "candidate_profile"


class ExperimentConfig(BaseModel):
    variants: dict[str, float]

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ExperimentConfig:
        total = sum(self.variants.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"experiment weights must sum to 1.0, got {total}")
        return self


class AppConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    collections: CollectionsConfig = Field(default_factory=CollectionsConfig)
    experiments: dict[str, ExperimentConfig] = Field(default_factory=dict)


# ---------------------------------------------------------------------- Companies
class CompanyEntry(BaseModel):
    name: str
    ats: Literal["greenhouse", "lever"]
    slug: str
    segment: Literal["product", "gcc", "startup", "services"] = "product"
    india_seen: bool = False
    enabled: bool = True


class AdapterToggle(BaseModel):
    enabled: bool = True
    endpoint: str


class CompaniesConfig(BaseModel):
    adapters: dict[str, AdapterToggle]
    refresh_cron: str = "0 6 * * *"
    location_filter: list[str] = Field(default_factory=list)
    companies: list[CompanyEntry] = Field(default_factory=list)

    def active(self, ats: str) -> list[CompanyEntry]:
        """Enabled companies for an enabled adapter."""
        toggle = self.adapters.get(ats)
        if toggle is None or not toggle.enabled:
            return []
        return [c for c in self.companies if c.ats == ats and c.enabled]


# ------------------------------------------------------------------------ Secrets
class Secrets(BaseModel):
    """Secrets read from the environment. Values are masked in repr and logs."""

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    azure_api_key: SecretStr | None = None
    adzuna_app_id: SecretStr | None = None
    adzuna_app_key: SecretStr | None = None
    qdrant_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    jwt_secret: SecretStr | None = None
    mcp_bearer_token: SecretStr | None = None

    ENV_MAP: ClassVar[dict[str, str]] = {
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "gemini_api_key": "GEMINI_API_KEY",
        "azure_api_key": "AZURE_API_KEY",
        "adzuna_app_id": "ADZUNA_APP_ID",
        "adzuna_app_key": "ADZUNA_APP_KEY",
        "qdrant_api_key": "QDRANT_API_KEY",
        "langfuse_public_key": "LANGFUSE_PUBLIC_KEY",
        "langfuse_secret_key": "LANGFUSE_SECRET_KEY",
        "jwt_secret": "APP_JWT_SECRET",
        "mcp_bearer_token": "MCP_SHARED_BEARER_TOKEN",
    }

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Secrets:
        return cls(**{field: env.get(var) for field, var in cls.ENV_MAP.items() if env.get(var)})

    @staticmethod
    def is_dummy(value: SecretStr | None) -> bool:
        """True for unset values and the placeholders shipped in .env.example."""
        return value is None or "dummy" in value.get_secret_value().lower()


# ----------------------------------------------------------------------- Settings
class Settings(BaseModel):
    env: Literal["dev", "staging", "prod"] = "dev"
    config_dir: Path = Path("config")
    prompts_dir: Path = Path("prompts")
    output_dir: Path = Path("outputs")
    vector_store: Literal["qdrant", "memory"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    langfuse_host: str = "http://localhost:3001"
    tracing_enabled: bool = False
    adzuna_country: str = "in"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    llm: LLMConfig
    app: AppConfig = Field(default_factory=AppConfig)
    companies: CompaniesConfig
    secrets: Secrets = Field(default_factory=Secrets)

    def assert_production_ready(self) -> None:
        """Fail fast if prod is started with placeholder credentials."""
        if self.env != "prod":
            return
        required = ["jwt_secret"]
        if self.llm.provider_profile in {"anthropic", "openai", "gemini", "azure"}:
            required.append(f"{self.llm.provider_profile}_api_key")
        bad = [name for name in required if Secrets.is_dummy(getattr(self.secrets, name))]
        if bad:
            raise ConfigError(f"production started with dummy or missing secrets: {bad}")


# ------------------------------------------------------------------------ Loading
def _read_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env reader (KEY=VALUE, '#' comments). Real env vars win."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_settings(
    root: Path | str = ".",
    env: Mapping[str, str] | None = None,
    dotenv: bool = True,
) -> Settings:
    """Build Settings from ``config/*.yaml`` plus environment variables.

    ``env`` defaults to ``os.environ``; tests pass an explicit mapping.
    """
    root = Path(root)
    merged: dict[str, str] = _load_dotenv(root / ".env") if dotenv else {}
    merged.update(env if env is not None else os.environ)

    config_dir = root / merged.get("APP_CONFIG_DIR", "config")
    llm_raw = _read_yaml(config_dir / "llm.yaml")
    if merged.get("LLM_PROVIDER_PROFILE"):
        llm_raw["provider_profile"] = merged["LLM_PROVIDER_PROFILE"]
    if merged.get("EMBEDDING_PROFILE"):
        llm_raw["embedding_profile"] = merged["EMBEDDING_PROFILE"]

    app_path = config_dir / "app.yaml"
    app_raw = _read_yaml(app_path) if app_path.exists() else {}

    settings = Settings(
        env=merged.get("APP_ENV", "dev"),  # type: ignore[arg-type]
        config_dir=config_dir,
        prompts_dir=root / merged.get("APP_PROMPTS_DIR", "prompts"),
        output_dir=root / merged.get("APP_OUTPUT_DIR", "outputs"),
        vector_store=merged.get("VECTOR_STORE", "qdrant"),  # type: ignore[arg-type]
        qdrant_url=merged.get("QDRANT_URL", "http://localhost:6333"),
        langfuse_host=merged.get("LANGFUSE_HOST", "http://localhost:3001"),
        tracing_enabled=merged.get("TRACING_ENABLED", "false").lower() == "true",
        adzuna_country=merged.get("ADZUNA_COUNTRY", "in"),
        cors_origins=[o.strip() for o in merged.get("APP_CORS_ORIGINS", "").split(",") if o.strip()]
        or ["http://localhost:3000"],
        llm=LLMConfig.model_validate(llm_raw),
        app=AppConfig.model_validate(app_raw),
        companies=CompaniesConfig.model_validate(_read_yaml(config_dir / "companies.yaml")),
        secrets=Secrets.from_env(merged),
    )
    settings.assert_production_ready()
    return settings
