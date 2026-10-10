from __future__ import annotations

from copilot.adapters.embeddings.hashing import HashingEmbedder
from copilot.adapters.embeddings.ollama import OllamaEmbedder
from copilot.adapters.embeddings.openai import OpenAIEmbedder
from copilot.config.settings import LLMConfig
from copilot.ports.embeddings import Embedder


def build_embedder(config: LLMConfig) -> Embedder:
    profile = config.active_embedding
    name = config.embedding_profile
    if name == "hashing":
        return HashingEmbedder(profile.dimensions)
    if name == "openai":
        return OpenAIEmbedder(profile)
    return OllamaEmbedder(profile)
